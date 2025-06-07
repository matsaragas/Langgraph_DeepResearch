from typing import TypedDict
from langgraph.graph import add_messages
from typing_extensions import Annotated
import operator

class OverallState(TypedDict):
    # messages have the type "list". `add_messages` function in the annotation
    # defines how this state key should be updated. Here it appends messages to
    # the list, rather than overwriting them
    messages: Annotated[list, add_messages]
    # it appends new lists to the existing list via concatenation (+)
    search_query: Annotated[list, operator.add]
    web_search_results = Annotated[list, operator.add]
    sources_gathered = Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class Query(TypedDict):
    query: str
    rational: str


class QueryGenerationState(TypedDict):
    query_list: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int








