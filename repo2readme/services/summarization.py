import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from repo2readme.utils.detect_language import detect_lang
from repo2readme.utils.workers import resolve_worker_count
from repo2readme.summarize.summary import summarize_file
from repo2readme.cache import SummaryCache
from repo2readme.services.reporting import SummaryFailure, is_failed_summary

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
    """
    total_documents = len(documents)
    summaries = []
    errors: List[SummaryFailure] = []

    if total_documents == 0:
        return summaries, errors

    summaries_lock = threading.Lock()
    errors_lock = threading.Lock()
    
    def process_document(doc):
        meta = doc["metadata"]
        file_path = meta["file_path"]
        try:
            lang = resolve_language(meta, doc["content"])
            cached = summary_cache.get(file_path, doc["content"], lang)
            if cached is not None:
                with summaries_lock:
                    summaries.append(cached)
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
            with summaries_lock:
                summaries.append(summary)
            # Only cache successful summaries; failed ones will be retried
            if not isinstance(summary, dict) or "error" not in summary:
                summary_cache.put(
                    file_path, doc["content"], lang, summary, meta.get("mtime", 0)
                )
        except Exception as e:
            with errors_lock:
                errors.append(SummaryFailure(file_path=file_path, reason=str(e)))


    effective_workers = resolve_worker_count(max_workers, total_documents)

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(process_document, doc): doc for doc in documents}
        
        for future in as_completed(futures):
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
                
    return summaries, errors

def build_directory_tree(file_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Builds a tree structure of the repository based on file paths.
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

# Below this many files the roll-up is skipped: the summaries already fit in one
# prompt, and a directory summary of a handful of files loses more detail than
# it saves context.
ROLLUP_THRESHOLD = 15


def generate_hierarchical_summaries(
    file_summaries: List[Dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    progress=None,
    task_id=None
) -> tuple[List[Dict[str, Any]], List[SummaryFailure]]:
    """
    Roll file summaries up into directory summaries, recursively.

    Returns the summaries to hand to the README prompt together with the
    directories whose roll-up failed. ``summarize_directory`` does not raise:
    it logs and returns a ``{"file_path": ..., "error": ...}`` placeholder,
    exactly as ``summarize_file`` does. Unlike a file placeholder, which the CLI
    partitions out before it reaches anything, a directory placeholder used to
    be returned as *the* summary of that directory - so a single failed call
    discarded every summary underneath it, and the ones underneath that, all the
    way to the leaves.

    A failed roll-up now falls back to the contents it was asked to condense.
    That is longer than the summary would have been, and it is what the run
    already paid for.
    """
    failures: List[SummaryFailure] = []

    if len(file_summaries) <= ROLLUP_THRESHOLD:
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
        return file_summaries, failures

    tree = build_directory_tree(file_summaries)

    from repo2readme.summarize.directory_summary import summarize_directory

    def count_dirs(node):
        return 1 + sum(count_dirs(c) for c in node["children"].values())

    total_dirs = count_dirs(tree) - 1 # excluding root
    if progress and task_id is not None:
        progress.update(task_id, total=total_dirs, completed=0)

    def process_dir(node: Dict[str, Any]) -> Any:
        child_summaries = []
        for child_name, child_node in node["children"].items():
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

        if is_failed_summary(dir_summary):
            # Keep what the directory is made of. The parent flattens a list,
            # so these travel upwards and are condensed at the next level if
            # that call succeeds.
            failures.append(
                SummaryFailure(
                    file_path=node["path"],
                    reason=str(dir_summary.get("error")),
                )
            )
            return contents

        return dir_summary

    top_level_summaries = process_dir(tree)

    if top_level_summaries is None:
        # Nothing in the tree had a usable path. Returning [None] would put a
        # bare null in the README prompt.
        return [], failures

    if isinstance(top_level_summaries, list):
        return top_level_summaries, failures
    return [top_level_summaries], failures
