from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from repo2readme.llm.factory import create_llm
import json
import logging

logger = logging.getLogger(__name__)

DIR_PROMPT_TEMPLATE = """
You are an expert code analyst.
Your task is to generate a concise summary of a directory based on the summaries of its contents (files and subdirectories).

Directory Path: {dir_path}

Contents Summaries:
{contents}

### RULES (STRICT)
- Output MUST be valid JSON.
- No code fences like ```json or ```.
- Your summary should synthesize the purpose of the directory, its key components, and how they interact.
- Do NOT list every single file. Instead, group related functionality.

Your returned JSON MUST contain exactly:
{{
  "file_path": "{dir_path}",
  "description": "A comprehensive summary of the directory's purpose and functionality."
}}
Return ONLY JSON.
{format_instructions}
"""

def summarize_directory(dir_path, contents_summaries, provider=None, model_name=None, base_url=None):
    model = create_llm(
        provider=provider or "groq",
        model=model_name or "openai/gpt-oss-120b",
        base_url=base_url,
    )
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template=DIR_PROMPT_TEMPLATE,
        input_variables=["dir_path", "contents"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | model | parser
    try:
        contents_str = json.dumps(contents_summaries, indent=2)
        return chain.invoke({
            "dir_path": dir_path,
            "contents": contents_str
        })
    except Exception as e:
        logger.warning("Directory summary error for %s: %s", dir_path, e)
        return {"file_path": dir_path, "error": str(e)}
