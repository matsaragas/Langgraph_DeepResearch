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



