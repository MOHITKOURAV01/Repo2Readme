import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from repo2readme.utils.detect_language import detect_lang
from repo2readme.summarize.summary import summarize_file
from repo2readme.cache import SummaryCache

def generate_all_summaries(
    documents: List[Dict[str, Any]],
    summary_cache: SummaryCache,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_workers: int | None = None,
    progress=None,
    task_id=None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Concurrently generates summaries for all documents.
    """
    total_documents = len(documents)
    summaries = []
    errors = []
    
    if total_documents == 0:
        return summaries, errors

    summaries_lock = threading.Lock()
    errors_lock = threading.Lock()
    
    def process_document(doc):
        meta = doc["metadata"]
        file_path = meta["file_path"]
        try:
            lang = detect_lang(meta.get("file_type", "text"), doc["content"])
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
                summary_cache.put(file_path, doc["content"], lang, summary, meta.get("mtime", 0))
        except Exception as e:
            with errors_lock:
                errors.append(f"Error processing {file_path}: {e}")
                
    effective_workers = min(max_workers or 4, total_documents)
    
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
            
        return dir_summary
        
    top_level_summaries = process_dir(tree)
    if isinstance(top_level_summaries, list):
        return top_level_summaries
    return [top_level_summaries]
