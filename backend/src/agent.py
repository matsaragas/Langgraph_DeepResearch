from langgraph.graph import StateGraph, START, END
from configuration import Configuration
from state import OverallState

from nodes import generate_query
from nodes import web_research
from nodes import reflection, finalize_answer, continue_to_web_research, evaluate_search

builder = StateGraph(OverallState, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)


builder.add_edge(START, "generate_query")

#add conditional edge to continue with the search queries in a parallel branch
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
builder.add_edge("web_research", "reflection")
#evaluate the search
builder.add_conditional_edges("reflection", evaluate_search, ["web_research", "finalize_answer"])
builder.add_edge("finalize_answer", END)
graph = builder.compile(name="pro-search-agent")





