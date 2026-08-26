import logging

from repo2readme.readme.agent_workflow import build_workflow
from repo2readme.readme.postprocess import postprocess_readme

logger = logging.getLogger(__name__)


class ReadmeGenerationError(RuntimeError):
    """The pipeline finished without producing a README worth writing.

    Raised instead of returning an empty string, because the caller's next step
    is to write the result over the user's ``README.md``.
    """


def run_pipeline(
    summaries: list,
    tree: str,
    dependency_overview: str,
    provider: str | None,
    model: str | None,
    base_url: str | None
) -> str:
    """
    Invokes the LangGraph workflow to generate the README and returns the result.

    The model's answer is normalized (wrapping code fence removed, trailing
    whitespace and blank line runs cleaned up) before being returned, and any
    structural problem that cannot be fixed mechanically - a table of contents
    pointing at a heading that no longer exists, a placeholder image - is
    logged as a warning rather than silently rewritten.

    Those same checks now run inside the loop as well, against each draft as it
    is produced, so what they find reaches the next generation round as feedback
    instead of only being reported once there is nothing left to do with it.
    See ``readme/agent_workflow.py``.

    Raises
    ------
    ReadmeGenerationError
        If the workflow produced no usable README. An empty result used to be
        returned like any other, and the CLI wrote it straight over the user's
        file while reporting success.
    """
    workflow = build_workflow()

    initial_state = {
        "summaries": summaries,
        "tree_structure": tree,
        "iteration_no": 0,
        "max_iterations": 3,
        'best_score': 0.0,
        "best_readme": "",
        # Structural problems in the draft currently being kept. Seeded so the
        # first comparison has something to compare against.
        "best_defects": 0,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "dependency_overview": dependency_overview,
    }

    final_state = workflow.invoke(initial_state)

    # Empty entries mark the iterations whose review came back normally.
    for error in final_state.get("review_errors") or []:
        if error:
            logger.warning("README review did not complete: %s", error)

    defects = final_state.get("defects") or []
    if len(defects) > 1:
        logger.info(
            "Structural problems per draft: %s",
            ", ".join(str(count) for count in defects),
        )

    readme, issues = postprocess_readme(select_readme(final_state))

    # The loop has already been told about these, on every round after the
    # first. Anything still here is what it could not fix, or what it never got
    # the chance to - the loop stops at max_iterations, and on a review failure.
    for issue in issues:
        logger.warning("README %s: %s", issue.kind, issue.message)

    if any(issue.kind == "empty" for issue in issues):
        raise ReadmeGenerationError(
            "The model returned no README content, so there is nothing to write."
        )

    return readme


def select_readme(final_state: dict) -> str:
    """Return the draft to keep from a finished workflow state.

    Normally the highest scoring draft. The last draft generated is the
    fallback: the scored draft is only recorded by the reviewer node, so
    anything that stops the graph before it runs would otherwise discard work
    that is perfectly usable.
    """
    best = (final_state.get("best_readme") or "").strip()
    if best:
        return final_state["best_readme"]

    drafts = final_state.get("readme") or []
    for draft in reversed(drafts):
        if draft and draft.strip():
            return draft

    return ""
