import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from repo2readme.readme.postprocess import (
    as_author_instructions,
    structural_findings,
)
from repo2readme.readme.readme_generator import generate_readme
from repo2readme.readme.reviewer_agent import readme_reviewer

logger = logging.getLogger(__name__)

# Score recorded for a draft whose review never came back. It has to be a real
# number because ``readme_condition`` reads the last score, and it has to be low
# enough never to win against a draft that was actually reviewed.
UNREVIEWED_SCORE = 0.0

# How close two reviewer scores have to be before the structural findings decide
# between them. The reviewer's score is a model's opinion on a 1-10 scale, so
# half a point apart is not a real difference; a broken table of contents is.
SCORE_TIE_TOLERANCE = 0.5


class ReadmeState(TypedDict):
    summaries: list[str]
    tree_structure: str
    readme: Annotated[list[str], operator.add]
    score: Annotated[list[float], operator.add]
    feedback: Annotated[list[str], operator.add]
    review_errors: Annotated[list[str], operator.add]
    #: Structural problems found in each draft, one entry per iteration.
    defects: Annotated[list[int], operator.add]
    best_readme: str
    best_score: float
    #: Structural problems in ``best_readme``, so a later draft can be compared
    #: against it without re-checking a string that is no longer around.
    best_defects: int
    iteration_no: int
    max_iterations: int
    provider: str | None
    model: str | None
    base_url: str | None
    dependency_overview: str


def choose_best(
    best_readme: str,
    best_score: float,
    candidate: str,
    score: float,
    best_defects: int = 0,
    candidate_defects: int = 0,
) -> tuple[str, float]:
    """Pick the draft to keep.

    A higher score wins, but an empty ``best_readme`` always loses: the loop
    starts with no draft and a best score of ``0.0``, so a first draft the
    reviewer scored ``0`` used to lose against nothing at all and the run ended
    up with an empty README.

    When the two scores are within :data:`SCORE_TIE_TOLERANCE` of each other,
    the structural findings decide. A draft scored 8.6 with three broken anchors
    is not better than one scored 8.4 with none, and the score cannot see the
    difference - it is a model's opinion of the prose, while the anchor either
    matches a heading or it does not.

    With equal defect counts - which is what the default arguments give - this
    behaves exactly as it did before.
    """
    if not best_readme.strip():
        return candidate, score

    if candidate_defects != best_defects and abs(score - best_score) <= SCORE_TIE_TOLERANCE:
        if candidate_defects < best_defects:
            return candidate, score
        return best_readme, best_score

    if score > best_score:
        return candidate, score
    return best_readme, best_score


def generate_readme_node(state: ReadmeState):
    latest_readme = (
        state["readme"][-1] if state.get("readme") else ""
    )
    readme = generate_readme(
        summaries=state['summaries'],
        tree_structure=state['tree_structure'],
        feedback=state['feedback'],
        latest_readme=latest_readme,
        provider=state["provider"],
        model_name=state["model"],
        base_url=state["base_url"],
        dependency_overview=state.get("dependency_overview", "")
    )

    return {
        'readme': [readme]
    }


def combined_feedback(review_feedback: str, instructions: str) -> str:
    """The reviewer's prose plus the structural findings, as one message.

    Both halves go into the same channel because that is what the generator
    reads; keeping them separate would mean changing the prompt as well.
    """
    parts = [part for part in (review_feedback or "", instructions) if part.strip()]
    return "\n\n".join(parts)


def readme_reviewer_node(state: ReadmeState):
    latest_readme = state['readme'][-1]

    # Checked before the review call, and independently of whether it comes
    # back: these findings are exact and cost nothing, so a failed review should
    # not also lose them.
    findings = structural_findings(latest_readme)
    instructions = as_author_instructions(findings)
    defects = len(findings)

    try:
        review = readme_reviewer(
            latest_readme,
            provider=state["provider"],
            model_name=state["model"],
            base_url=state["base_url"],
        )
    except Exception as exc:
        # The reviewer is an improvement step, not a gate. Losing it costs the
        # run another polishing round; it must not cost the run the draft that
        # every summary was spent on, which is what re-raising here did.
        logger.warning("README review failed, keeping the current draft: %s", exc)
        best_readme, best_score = choose_best(
            state['best_readme'],
            state['best_score'],
            latest_readme,
            UNREVIEWED_SCORE,
            best_defects=state.get('best_defects', 0),
            candidate_defects=defects,
        )
        return {
            'score': [UNREVIEWED_SCORE],
            'feedback': [instructions],
            'review_errors': [str(exc)],
            'defects': [defects],
            'iteration_no': state['iteration_no'] + 1,
            'best_readme': best_readme,
            'best_score': best_score,
            'best_defects': _defects_of(
                best_readme, latest_readme, defects, state
            ),
        }

    best_readme, best_score = choose_best(
        state['best_readme'],
        state['best_score'],
        latest_readme,
        review.score,
        best_defects=state.get('best_defects', 0),
        candidate_defects=defects,
    )

    if findings:
        logger.info(
            "Draft %d has %d structural problem(s); telling the next round about them",
            state['iteration_no'] + 1,
            defects,
        )

    return {
        'score': [review.score],
        'feedback': [combined_feedback(review.feedback, instructions)],
        # One entry per iteration, empty when the review came back, so the
        # latest entry always describes the latest review. Returning [] here
        # would add nothing to an accumulating channel, leaving an older
        # failure looking like the current one.
        'review_errors': [''],
        'defects': [defects],
        'iteration_no': state['iteration_no'] + 1,
        'best_score': best_score,
        'best_readme': best_readme,
        'best_defects': _defects_of(best_readme, latest_readme, defects, state),
    }


def _defects_of(
    best_readme: str, candidate: str, candidate_defects: int, state: ReadmeState
) -> int:
    """Defect count that goes with whichever draft was kept."""
    if best_readme == candidate:
        return candidate_defects
    return state.get('best_defects', 0)


def latest_review_error(state: ReadmeState) -> str:
    """Why the most recent review failed, or ``""`` if it came back."""
    errors = state.get('review_errors') or ['']
    return errors[-1]


def readme_condition(state: ReadmeState):
    score = state['score'][-1]
    max_iterations = state['max_iterations']
    iteration = state['iteration_no']

    # Without a review there is no feedback to improve on, so another round
    # would just re-ask the same question and spend another call. Only the
    # latest review counts: an earlier failure must not end a run whose most
    # recent review succeeded.
    if latest_review_error(state):
        return END

    if score >= 8.5 or iteration >= max_iterations:
        return END
    else:
        return 'generate_readme'


def build_workflow():
    """Build and compile the LangGraph workflow on demand."""
    graph = StateGraph(ReadmeState)
    graph.add_node('generate_readme', generate_readme_node)
    graph.add_node('readme_reviewer', readme_reviewer_node)

    graph.add_edge(START, 'generate_readme')
    graph.add_edge('generate_readme', 'readme_reviewer')
    graph.add_conditional_edges(
        'readme_reviewer',
        readme_condition,
        {END: END, "generate_readme": "generate_readme"},
    )

    return graph.compile()
