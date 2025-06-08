
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage
from langgraph.types import Send

from state import WebSearchState, ReflectionState, QueryGenerationState, OverallState


from configuration import Configuration
from tools_and_schemas import SearchQueryList, Reflection
from utils import get_current_date, resolve_urls, get_citations, insert_citation_markers

from prompts import (reflection_instructions,
                     answer_instructions,
                     web_research_instructions,
                     query_writer_instructions)


from utils import get_research_topic

from langchain_openai import ChatOpenAI

from google.genai import Client
import os


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search query based on the User's questions
    Args:
        state: Current Graph state containing user's question
        config: Configuration for the runnable, including LLM provider settings
    """
    configurable = Configuration.from_runnable_config(config)
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key="API_KEY"
    )
    structured_llm = llm.with_structured_output(SearchQueryList)
    current_date = get_current_date()

    # Format the prompt
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"]
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]

def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API.
    Args:
         state: Current graph state containing the search query and research loop count
         config: Configuration for the runnable, include search API settings

    Returns:
        Dictionary with state update, including sources gathered, research_loop_count, and
        web_research_results
    """
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_research_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"]
    )

    genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )

    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    LangGraph node that identifies knowledge gaps and generates potential follow-up series.
    :param state:
    :param config:
    :return:
    """
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"])
    )
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key="API_KEY"
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": result.research_loop_count,
        "number_of_ran_queries": len(state["search_query"])
    }


def evaluate_search(state: ReflectionState, config: RunnableConfig,
) -> OverallState:

    """
    Langgraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information or
    to finalize the summary based on the configured maximim number of research loops
    :param state:
    :param config:
    :return: A String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_reseach_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx)
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Langgraph node that finalizes the research query
    Prepares the final output by de-duplicating and formatting sources, then
    combining them with the running summary to create a well-structured research
    report with proper citations.
    Args:
        state: Current graph state containing the running summary and sources gathered.

    Returns:
        Dictionaries with state update, including running_summary key containing the
        formatted final summary with sources.
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key="API_KEY"
    )
    result = llm.invoke(formatted_prompt)

    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }















