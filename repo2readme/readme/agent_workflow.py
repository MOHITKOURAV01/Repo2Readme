import logging
import operator
from typing import Annotated, List, TypedDict

from langgraph.graph import END, START, StateGraph

from repo2readme.readme.readme_generator import generate_readme
from repo2readme.readme.reviewer_agent import readme_reviewer

logger = logging.getLogger(__name__)

# Score recorded for a draft whose review never came back. It has to be a real
# number because ``readme_condition`` reads the last score, and it has to be low
# enough never to win against a draft that was actually reviewed.
UNREVIEWED_SCORE = 0.0


class ReadmeState(TypedDict):
    summaries: List[str]
    tree_structure: str
    readme: Annotated[list[str], operator.add]
    score: Annotated[list[float], operator.add]
    feedback: Annotated[list[str], operator.add]
    review_errors: Annotated[list[str], operator.add]
    best_readme: str
    best_score: float
    iteration_no: int
    max_iterations: int
    provider: str | None
    model: str | None
    base_url: str | None
    dependency_overview: str


def choose_best(
    best_readme: str, best_score: float, candidate: str, score: float
) -> tuple[str, float]:
    """Pick the draft to keep.

    A higher score wins, but an empty ``best_readme`` always loses: the loop
    starts with no draft and a best score of ``0.0``, so a first draft the
    reviewer scored ``0`` used to lose against nothing at all and the run ended
    up with an empty README.
    """
    if not best_readme.strip():
        return candidate, score
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


def readme_reviewer_node(state: ReadmeState):
    latest_readme = state['readme'][-1]

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
            state['best_readme'], state['best_score'], latest_readme, UNREVIEWED_SCORE
        )
        return {
            'score': [UNREVIEWED_SCORE],
            'feedback': [''],
            'review_errors': [str(exc)],
            'iteration_no': state['iteration_no'] + 1,
            'best_readme': best_readme,
            'best_score': best_score,
        }

    best_readme, best_score = choose_best(
        state['best_readme'], state['best_score'], latest_readme, review.score
    )

    return {
        'score': [review.score],
        'feedback': [review.feedback],
        'review_errors': [],
        'iteration_no': state['iteration_no'] + 1,
        'best_score': best_score,
        'best_readme': best_readme,
    }


def readme_condition(state: ReadmeState):
    score = state['score'][-1]
    max_iterations = state['max_iterations']
    iteration = state['iteration_no']

    # Without a review there is no feedback to improve on, so another round
    # would just re-ask the same question and spend another call.
    if state.get('review_errors'):
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
