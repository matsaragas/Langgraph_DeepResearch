from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web search"
    )
    rational: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic"
    )

