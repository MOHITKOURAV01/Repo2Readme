"""The review loop should be told what the structural checks already know.

From the issue: ``postprocess_readme`` finds broken table-of-contents anchors,
placeholder images and missing or duplicated top-level headings - exactly, from
the string, for free. It ran after ``workflow.invoke()`` had returned, so the
findings were logged and the README was written anyway, while the loop had spent
up to three generation calls and three review calls with no idea any of it was
wrong.
"""

import logging

import pytest

from repo2readme.readme import agent_workflow
from repo2readme.readme.agent_workflow import (
    SCORE_TIE_TOLERANCE,
    UNREVIEWED_SCORE,
    choose_best,
    combined_feedback,
    readme_reviewer_node,
)
from repo2readme.readme.postprocess import (
    REMEDIES,
    ValidationIssue,
    as_author_instructions,
    structural_findings,
)

CLEAN = """# Project

## Table of Contents
- [Usage](#usage)

## Usage

Run it.
"""

BROKEN_ANCHOR = """# Project

## Table of Contents
- [Configuration](#configuration)

## Usage

Run it.
"""

PLACEHOLDER_IMAGE = """# Project

![logo](path/to/logo.png)

## Usage

Run it.
"""

NO_HEADING = """Just a paragraph with no heading at all.
"""


class Review:
    def __init__(self, score, feedback="tighten the intro"):
        self.score = score
        self.feedback = feedback


def state(readme, **overrides):
    base = {
        "readme": [readme],
        "best_readme": "",
        "best_score": 0.0,
        "best_defects": 0,
        "iteration_no": 0,
        "provider": None,
        "model": None,
        "base_url": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# structural_findings / as_author_instructions
# ---------------------------------------------------------------------------


def test_a_clean_draft_has_no_findings():
    assert structural_findings(CLEAN) == []
    assert as_author_instructions([]) == ""


@pytest.mark.parametrize(
    "draft,kind",
    [
        (BROKEN_ANCHOR, "broken-anchor"),
        (PLACEHOLDER_IMAGE, "placeholder-image"),
        (NO_HEADING, "missing-h1"),
        ("# One\n\n# Two\n", "duplicate-h1"),
        ("   \n", "empty"),
    ],
)
def test_each_kind_is_found(draft, kind):
    assert kind in [issue.kind for issue in structural_findings(draft)]


def test_a_draft_wrapped_in_a_code_fence_is_unwrapped_first():
    """Otherwise a fenced answer looks like a document with no headings."""
    assert structural_findings("```markdown\n" + CLEAN + "```\n") == []


def test_instructions_name_the_offending_anchor():
    instructions = as_author_instructions(structural_findings(BROKEN_ANCHOR))
    assert "#configuration" in instructions
    assert REMEDIES["broken-anchor"] in instructions


def test_instructions_say_what_to_do_about_each_kind():
    issues = [
        ValidationIssue(kind=kind, message=f"message for {kind}")
        for kind in REMEDIES
    ]
    instructions = as_author_instructions(issues)
    for kind, remedy in REMEDIES.items():
        assert f"message for {kind}" in instructions
        assert remedy in instructions


def test_an_unknown_kind_still_renders_its_message():
    instructions = as_author_instructions(
        [ValidationIssue(kind="something-new", message="a new problem")]
    )
    assert "a new problem" in instructions


# ---------------------------------------------------------------------------
# combined_feedback
# ---------------------------------------------------------------------------


def test_combined_feedback_keeps_both_halves():
    combined = combined_feedback("prose", "structure")
    assert "prose" in combined
    assert "structure" in combined


@pytest.mark.parametrize(
    "review,instructions,expected",
    [
        ("prose", "", "prose"),
        ("", "structure", "structure"),
        ("   ", "", ""),
        (None, "", ""),
    ],
)
def test_combined_feedback_drops_empty_halves(review, instructions, expected):
    assert combined_feedback(review, instructions) == expected


# ---------------------------------------------------------------------------
# choose_best
# ---------------------------------------------------------------------------


def test_defect_free_draft_wins_a_near_tie():
    kept, score = choose_best(
        "with defects", 8.6, "clean", 8.4, best_defects=3, candidate_defects=0
    )
    assert (kept, score) == ("clean", 8.4)


def test_a_clearly_better_score_still_wins_despite_defects():
    kept, _ = choose_best(
        "clean", 5.0, "flawed", 9.0, best_defects=0, candidate_defects=2
    )
    assert kept == "flawed"


def test_a_defective_candidate_does_not_win_a_near_tie():
    kept, score = choose_best(
        "clean", 8.4, "flawed", 8.6, best_defects=0, candidate_defects=2
    )
    assert (kept, score) == ("clean", 8.4)


def test_the_tie_window_is_the_documented_one():
    outside = SCORE_TIE_TOLERANCE + 0.1
    kept, _ = choose_best(
        "clean", 8.0, "flawed", 8.0 + outside, best_defects=0, candidate_defects=2
    )
    assert kept == "flawed"


def test_equal_defects_behaves_exactly_as_before():
    assert choose_best("", 0.0, "draft", 0.0) == ("draft", 0.0)
    assert choose_best("a", 4.0, "b", 7.5) == ("b", 7.5)
    assert choose_best("b", 7.5, "a", 2.0) == ("b", 7.5)
    assert choose_best("a", 7.5, "b", 7.5) == ("a", 7.5)
    assert choose_best("   \n ", 9.9, "draft", 1.0) == ("draft", 1.0)


def test_an_empty_best_still_always_loses():
    kept, _ = choose_best("", 9.9, "flawed", 1.0, best_defects=0, candidate_defects=5)
    assert kept == "flawed"


# ---------------------------------------------------------------------------
# The reviewer node
# ---------------------------------------------------------------------------


def test_findings_reach_the_next_round_as_feedback(monkeypatch):
    monkeypatch.setattr(
        agent_workflow, "readme_reviewer", lambda *a, **k: Review(7.0)
    )

    result = readme_reviewer_node(state(BROKEN_ANCHOR))

    assert result["defects"] == [1]
    assert "tighten the intro" in result["feedback"][0]
    assert "#configuration" in result["feedback"][0]


def test_a_clean_draft_adds_nothing_to_the_feedback(monkeypatch):
    monkeypatch.setattr(
        agent_workflow, "readme_reviewer", lambda *a, **k: Review(7.0)
    )

    result = readme_reviewer_node(state(CLEAN))

    assert result["defects"] == [0]
    assert result["feedback"] == ["tighten the intro"]


def test_findings_survive_a_failed_review(monkeypatch):
    """A review that never came back should not also lose the free checks."""

    def explode(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(agent_workflow, "readme_reviewer", explode)

    result = readme_reviewer_node(state(BROKEN_ANCHOR))

    assert result["review_errors"] == ["provider down"]
    assert result["score"] == [UNREVIEWED_SCORE]
    assert result["defects"] == [1]
    assert "#configuration" in result["feedback"][0]
    # And the draft is still kept, as it was before.
    assert result["best_readme"] == BROKEN_ANCHOR


def test_the_kept_draft_carries_its_own_defect_count(monkeypatch):
    monkeypatch.setattr(
        agent_workflow, "readme_reviewer", lambda *a, **k: Review(9.0)
    )

    result = readme_reviewer_node(state(BROKEN_ANCHOR))
    assert result["best_readme"] == BROKEN_ANCHOR
    assert result["best_defects"] == 1

    # A later, worse draft does not overwrite the count of the one being kept.
    monkeypatch.setattr(
        agent_workflow, "readme_reviewer", lambda *a, **k: Review(2.0)
    )
    later = readme_reviewer_node(
        state(
            PLACEHOLDER_IMAGE,
            best_readme=BROKEN_ANCHOR,
            best_score=9.0,
            best_defects=1,
        )
    )
    assert later["best_readme"] == BROKEN_ANCHOR
    assert later["best_defects"] == 1


def test_the_loop_prefers_the_clean_draft_over_a_marginally_better_score(monkeypatch):
    monkeypatch.setattr(
        agent_workflow, "readme_reviewer", lambda *a, **k: Review(8.4)
    )

    result = readme_reviewer_node(
        state(CLEAN, best_readme=BROKEN_ANCHOR, best_score=8.6, best_defects=1)
    )

    assert result["best_readme"] == CLEAN
    assert result["best_defects"] == 0


def test_the_stopping_rule_is_unchanged(monkeypatch):
    """Structural findings must not buy the run extra API calls."""
    from langgraph.graph import END

    from repo2readme.readme.agent_workflow import readme_condition

    ended = readme_condition(
        {
            "score": [9.0],
            "max_iterations": 3,
            "iteration_no": 1,
            "review_errors": [""],
            "defects": [4],
        }
    )
    assert ended == END


def test_a_run_logs_the_defect_counts(monkeypatch, caplog):
    from repo2readme.services import orchestrator

    class Workflow:
        def invoke(self, initial_state):
            return {
                "best_readme": CLEAN,
                "readme": [BROKEN_ANCHOR, CLEAN],
                "review_errors": ["", ""],
                "defects": [1, 0],
            }

    monkeypatch.setattr(orchestrator, "build_workflow", lambda: Workflow())

    with caplog.at_level(logging.INFO):
        readme = orchestrator.run_pipeline(
            summaries=[], tree="", dependency_overview="",
            provider=None, model=None, base_url=None,
        )

    assert readme.startswith("# Project")
    assert "Structural problems per draft: 1, 0" in caplog.text


def test_the_initial_state_seeds_the_defect_count(monkeypatch):
    from repo2readme.services import orchestrator

    seen = {}

    class Workflow:
        def invoke(self, initial_state):
            seen.update(initial_state)
            return {"best_readme": CLEAN, "readme": [CLEAN]}

    monkeypatch.setattr(orchestrator, "build_workflow", lambda: Workflow())
    orchestrator.run_pipeline(
        summaries=[], tree="", dependency_overview="",
        provider=None, model=None, base_url=None,
    )

    assert seen["best_defects"] == 0
