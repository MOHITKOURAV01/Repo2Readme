from langchain_core.prompts import PromptTemplate

import hashlib
import logging
import os
from langchain_core.output_parsers import JsonOutputParser
from repo2readme.llm.factory import create_llm
from repo2readme.utils.content_budget import annotate_truncation, apply_content_budget
from repo2readme.utils.retry import call_with_retry



logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are an expert code analyst.
Your task is generate a summary of the code that summary will help later to 
generate a README.md file. 
The code file can be in any programming language.So be careful about with which
file type are you dealing with. 

---
### RULES (STRICT)
- Output MUST be valid JSON.
- No code fences like ```json or ```.
- No trailing commas.
- No multiline strings. Use \\n instead.
- Arrays may contain:
  - simple strings
  - or JSON objects {{"name": "...", "description": "..."}}
- You may add ANY extra fields, but every value must be valid JSON.
- DO NOT guess missing information.
- DO NOT rewrite the code.
- DO NOT hallucinate what the code doesn't contain.

---
Your summary MUST include:
- File path 
{file_path}
- Purpose of the code in the file
- If the line of code is less the summary will be short
and the if the line of code is high the summary will be large.
- The summary size will depend on the total line of codes
- Add key functions, classes, and their purpose
- Important logic and algorithms
- All the dependencies and integrations with other files
- Any configuration, environment variables, or API usage
- Also add the information if the code has any important portions
---


Your returned JSON MUST contain at least:
{{
  "file_path": "{file_path}",
  "description": ""
}}


You may include other fields if the code contains relevant information
(e.g., functions, classes, logic, dependencies, configs, etc.)

Now summarize this code:

File Path: {file_path}
Language: {language}

Code:
{content}
Return ONLY JSON.  
{format_instructions}
"""


def get_prompt_template_hash() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def create_summarizer(file_path, language,
    content,provider=None, model_name=None, base_url=None,):
    model = create_llm(
        provider=provider or "groq",
        model=model_name,
        base_url=base_url,
    )
    parser=JsonOutputParser()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["file_path", "language", "content"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    return chain


def summarize_file(
    file_path,
    language,
    content,
    provider=None,
    model_name=None,
    base_url=None,
    max_content_chars=None,
):
    """Summarize one file, sending at most ``max_content_chars`` of it.

    A file just under ``--max-file-size-kb`` is 60-70 thousand tokens in one
    request, which the provider rejects as too large - a rejection the retry
    correctly refuses to repeat, so the file was dropped from the run after
    being paid for. It is cut to the budget instead, head and tail, with the
    gap marked, and the summary records that it describes an excerpt.
    """
    budgeted = apply_content_budget(content, max_content_chars)

    if budgeted.truncated:
        logger.info(
            "%s is %d lines; sending the first and last %d, omitting %d",
            file_path,
            budgeted.original_lines,
            budgeted.original_lines - budgeted.omitted_lines,
            budgeted.omitted_lines,
        )

    try:
        chain = create_summarizer(
            file_path, language, budgeted.text, provider, model_name, base_url,
        )
        summary = call_with_retry(
            lambda: chain.invoke({
                "file_path": file_path.replace("\\", "/"),
                "language": language,
                "content": budgeted.text
            }),
            description=f"summary for {file_path}",
        )
        return annotate_truncation(summary, budgeted)
    except Exception as e:
        logger.warning("Summary error for %s: %s", file_path, e)
        return {"file_path": file_path, "error": str(e)}
