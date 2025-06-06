from langgraph.graph import StateGraph
from backend.src.configuration import Configuration
from backend.src.state import OverallState

from backend.src.nodes import generate_query
from backend.src.nodes import web_research
from backend.src.nodes import reflection

builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
