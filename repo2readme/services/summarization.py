from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from repo2readme.utils.detect_language import detect_lang
from repo2readme.utils.workers import resolve_worker_count
from repo2readme.summarize.summary import summarize_file
from repo2readme.cache import SummaryCache
from repo2readme.services.reporting import SummaryFailure

# Marks a slot that was never filled - the document failed, so it contributes
# nothing to the summaries. A sentinel rather than None, because a summary is
# whatever the provider's chain returned and None is a value it may legitimately
# produce.
_UNFILLED = object()

def resolve_language(metadata: Dict[str, Any], content: str) -> str:
    """The language of a document, preferring what the traversal already found.

    The traversal pipeline detects the language from the full path *and* the
    content, so it can use every strategy ``detect_lang`` implements - the
    extension, the filename, a shebang, then content markers. That answer used
    to be dropped before it reached here, and this stage re-detected it from
    ``file_type``, which is the bare extension and empty for any file without
    one. A ``Gemfile`` came out as ``unknown`` and a ``Jenkinsfile`` as
    ``json``, and the wrong answer went into the summarizer prompt and the
    cache key.

    The fallback is kept for callers that build documents by hand, and is given
    the best path it can find rather than the extension alone.
    """
    language = metadata.get("language")
    if language and language != "unknown":
        return language

    path = (
        metadata.get("relative_path")
        or metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("file_type")
        or "text"
    )
    return detect_lang(path, content)


def generate_all_summaries(
    documents: List[Dict[str, Any]],
    summary_cache: SummaryCache,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_workers: int | None = None,
    progress=None,
    task_id=None
) -> tuple[List[Dict[str, Any]], List[SummaryFailure]]:
    """
    Concurrently generates summaries for all documents.

    Returns the summaries together with the files that raised before a summary
    could be produced. Summaries that came back as ``{"error": ...}``
    placeholders stay in the first list; use
    ``repo2readme.services.reporting.partition_summaries`` to split them out.

    Both lists come back in the order ``documents`` was given, whatever order the
    workers finished in. They used to be appended as each future completed, so
    two runs over the same repository - same provider, same model, even a fully
    warm cache - handed the README prompt a differently ordered list and got a
    different README back. The traversal pipeline already writes each document
    into a preallocated slot for exactly this reason; this stage does the same,
    so re-running the tool on an unchanged repository produces an unchanged
    README.
    """
    total_documents = len(documents)

    if total_documents == 0:
        return [], []

    # One slot per document, and each slot is written by exactly one worker, so
    # no lock is needed: the ordering comes from the index, not from when the
    # write happened.
    slots: List[Any] = [_UNFILLED] * total_documents
    failures: List[SummaryFailure | None] = [None] * total_documents

    def process_document(index: int, doc: Dict[str, Any]) -> None:
        meta = doc["metadata"]
        file_path = meta["file_path"]
        try:
            lang = resolve_language(meta, doc["content"])
            cached = summary_cache.get(file_path, doc["content"], lang)
            if cached is not None:
                slots[index] = cached
                return

            if provider or model or base_url:
                summary = summarize_file(
                    file_path=file_path,
                    language=lang,
                    content=doc["content"],
                    provider=provider,
                    model_name=model,
                    base_url=base_url,
                )
            else:
                summary = summarize_file(
                    file_path=file_path,
                    language=lang,
                    content=doc["content"],
                )
            slots[index] = summary
            # Only cache successful summaries; failed ones will be retried
            if not isinstance(summary, dict) or "error" not in summary:
                summary_cache.put(
                    file_path, doc["content"], lang, summary, meta.get("mtime", 0)
                )
        except Exception as e:
            failures[index] = SummaryFailure(file_path=file_path, reason=str(e))

    effective_workers = resolve_worker_count(max_workers, total_documents)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(process_document, index, doc)
            for index, doc in enumerate(documents)
        ]

        for _ in as_completed(futures):
            if progress and task_id is not None:
                progress.update(task_id, advance=1)

    summaries = [entry for entry in slots if entry is not _UNFILLED]
    errors = [failure for failure in failures if failure is not None]

    return summaries, errors

def build_directory_tree(file_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds a tree structure of the repository based on file paths.

    Files keep the order they were given in, which is repository order now that
    the summaries themselves are ordered. Directories are walked in sorted order
    by :func:`generate_hierarchical_summaries`, so the roll-up does not depend on
    which directory happened to be seen first.
    """
    tree = {"type": "dir", "path": ".", "files": [], "children": {}}
    for summary in file_summaries:
        if isinstance(summary, str):
            continue
        path = summary.get("file_path", "")
        if not path:
            continue
        parts = path.split("/")
        current = tree
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current["files"].append(summary)
            else:
                if part not in current["children"]:
                    current["children"][part] = {
                        "type": "dir",
                        "path": "/".join(parts[:i+1]),
                        "files": [],
                        "children": {}
                    }
                current = current["children"][part]
    return tree

def generate_hierarchical_summaries(
    file_summaries: List[Dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress=None,
    task_id=None
) -> List[Dict[str, Any]]:
    """
    Rolls up file summaries into directory summaries recursively.
    """
    if len(file_summaries) <= 15:
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        return file_summaries
        
    tree = build_directory_tree(file_summaries)
    
    from repo2readme.summarize.directory_summary import summarize_directory
    
    def count_dirs(node):
        return 1 + sum(count_dirs(c) for c in node["children"].values())
        
    total_dirs = count_dirs(tree) - 1 # excluding root
    if progress and task_id is not None:
        progress.update(task_id, total=total_dirs, completed=0)
    
    def process_dir(node: Dict[str, Any]) -> Any:
        child_summaries = []
        # Sorted, so a directory summary is built from the same contents in the
        # same order on every run.
        for child_name in sorted(node["children"]):
            child_node = node["children"][child_name]
            child_summary = process_dir(child_node)
            if child_summary:
                if isinstance(child_summary, list):
                    child_summaries.extend(child_summary)
                else:
                    child_summaries.append(child_summary)
                
        contents = child_summaries + node["files"]
        if not contents:
            return None
            
        if node["path"] == ".":
            return contents

        if len(contents) == 1:
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
            return contents[0]
            
        dir_summary = summarize_directory(
            dir_path=node["path"],
            contents_summaries=contents,
            provider=provider,
            model_name=model,
            base_url=base_url
        )
        
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
            
        return dir_summary
        
    top_level_summaries = process_dir(tree)
    if isinstance(top_level_summaries, list):
        return top_level_summaries
    return [top_level_summaries]
