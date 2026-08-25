from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
import os
from repo2readme.llm.factory import create_llm
from repo2readme.utils.retry import call_with_retry


class ReviewSchema(BaseModel):
    score:float=Field(description="A score between 1 and 10 based on the quality of the README")
    feedback:str=Field(description="Actionable comment how to improve the README")

REVIEW_PROMPT_TEMPLATE = """
You are a senior technical writer and software engineer acting as a README reviewer.

Your job is to Review the following README and do two tasks:
1. Give a SCORE from 1 to 10 (can be float) for the README.
2. Provide a actionable FEEDBACK for improvement.

## Consider the following criteria for the SCORE and FEEDBACK:
-**Clarity**:
- **Readability**
- **Structure**
- **Completeness**

Return ONLY JSON in this format:
{format_instructions}
## README to review:
{readme}



    """


def readme_reviewer(
    readme: str,
    provider: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
):
    model = create_llm(
        provider=provider or "google",
        model=model_name,
        base_url=base_url,
    )
    parser=PydanticOutputParser(pydantic_object=ReviewSchema)
    review_prompt=PromptTemplate(
        template=REVIEW_PROMPT_TEMPLATE,
    input_variables=['readme'],
    partial_variables={"format_instructions":parser.get_format_instructions()}
    )
    chain=review_prompt | model | parser
    response = call_with_retry(
        lambda: chain.invoke({'readme': readme}),
        description="README review",
    )
    return response
