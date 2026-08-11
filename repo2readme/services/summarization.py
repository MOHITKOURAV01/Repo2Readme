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
                
    max_workers = min(2, total_documents)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_document, doc): doc for doc in documents}
        
        for future in as_completed(futures):
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
                
    return summaries, errors
