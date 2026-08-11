from repo2readme.readme.agent_workflow import workflow

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
    """
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
    return final_state['best_readme']
