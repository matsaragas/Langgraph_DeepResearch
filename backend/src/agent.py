from langgraph.graph import StateGraph
from configuration import Configuration


builder = StateGraph(OverallState, config_schema=Configuration)