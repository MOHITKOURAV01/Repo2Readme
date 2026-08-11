def format_size(size_bytes: int) -> str:
    """Formats a byte size into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def estimate_analysis_cost(documents: list) -> tuple[int, int, int]:
    """
    Estimates the tokens, size in bytes, and number of documents.
    Returns: (estimated_tokens, total_size_bytes, total_documents)
    """
    estimated_tokens = sum(max(1, len(doc["content"]) // 3) for doc in documents)
    total_size_bytes = sum(len(doc["content"].encode("utf-8")) for doc in documents)
    total_documents = len(documents)
    return estimated_tokens, total_size_bytes, total_documents
