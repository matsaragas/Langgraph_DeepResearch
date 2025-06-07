from backend.src.state import OverallState
from langchain_core.runnables import RunnableConfig

from backend.src.state import QueryGeneratingState
from backend.src.state import WebSearchState


from backend.src.configuration import Configuration
from backend.src.tools_and_schemas import SearchQueryList, Reflection
from backend.src.utils import get_current_date, resolve_urls, get_citations

from backend.src.prompts import query_writer_instructions
from backend.src.prompts import web_research_instructions
from backend.src.prompts import reflection_instructions


from backend.src.utils import get_research_topic

from langchain_openai import ChatOpenAI

from google.genai import Client
import os


genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGeneratingState:
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













