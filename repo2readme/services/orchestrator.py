import logging

from repo2readme.readme.agent_workflow import build_workflow
from repo2readme.readme.postprocess import postprocess_readme

logger = logging.getLogger(__name__)

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
    """
    workflow = build_workflow()

    initial_state = {
        "summaries": summaries,
        "tree_structure": tree,
        "iteration_no": 0,
        "max_iterations": 3,
        "latest_readme": "",
        'best_score': 0.0,
        "best_readme": "",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "dependency_overview": dependency_overview,
    }

    final_state = workflow.invoke(initial_state)

    readme, issues = postprocess_readme(final_state['best_readme'])

    for issue in issues:
        logger.warning("README %s: %s", issue.kind, issue.message)

    return readme
