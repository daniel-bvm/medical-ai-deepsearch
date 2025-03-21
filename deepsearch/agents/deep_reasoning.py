from typing import List, Dict, Any, Tuple
import os
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from deepsearch.models import SearchState, SearchResult

# Set up logging
logger = logging.getLogger("deepsearch.deep_reasoning")

# Load environment variables
load_dotenv()

# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

# Define the prompt template for analysis and reasoning
REASONING_TEMPLATE = """You are an expert research analyst and reasoning agent. Your task is to analyze search results,
identify relevant information, and determine if further searches are needed.

ORIGINAL QUERY: {original_query}

CURRENT SEARCH ITERATION: {iteration}

SEARCH RESULTS:
{search_results}

INSTRUCTIONS:
1. Analyze the search results carefully to extract key information related to the original query.
2. Identify any knowledge gaps that require further searches.
3. Decide if the search process should continue or if we have sufficient information to answer the query.
4. If further searches are needed, generate specific new search queries to fill the knowledge gaps.
5. Format your response as a JSON object with the following structure:

{{
  "key_points": ["point 1", "point 2", "..."],
  "knowledge_gaps": ["gap 1", "gap 2", "..."],
  "new_queries": ["query 1", "query 2", "..."],
  "search_complete": true/false,
  "reasoning": "Your explanation of why the search is complete or needs to continue"
}}

Your response MUST be a valid JSON object with all the fields present. Keep each key point and knowledge gap concise.
If there are no knowledge gaps or the search should stop, return an empty array for "knowledge_gaps" and "new_queries"
and set "search_complete" to true.

IMPORTANT: If this is already iteration {max_iterations} or higher, set "search_complete" to true regardless of knowledge gaps.
"""

# Define the prompt template for final answer formulation
ANSWER_TEMPLATE = """You are a comprehensive research analyst tasked with providing a thorough answer to a query based on search results.

ORIGINAL QUERY: {original_query}

KEY POINTS FROM SEARCH RESULTS:
{key_points}

SEARCH DETAILS:
{search_details}

INSTRUCTIONS:
Your task is to formulate a complete answer with three distinct sections:

1. KEY POINTS: A bulleted list of the most important findings and facts (5-7 points max)
2. DIRECT ANSWER: A concise paragraph directly answering the original query
3. DETAILED NOTES: A comprehensive explanation with supporting evidence from the search results

Format your answer using proper markdown sections. Make sure each section is clearly defined and contains the appropriate level of detail.
Analyze the key points carefully to ensure your direct answer is accurate and addresses the core question.
In the detailed notes, be thorough but clear, organizing information logically and citing specific sources where relevant.

DO NOT include phrases like "Based on the search results" or "According to the information provided."
Just present the information directly and authoritatively.
"""

def init_reasoning_llm(temperature: float = 0.3):
    """Initialize the language model for reasoning using OpenAI-compatible API."""
    # Use OpenAI-compatible server
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "no-need"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=temperature,
        max_tokens=1024
    )
    return llm

def format_search_results(state: SearchState) -> str:
    """Format the search results for the prompt."""
    results_text = ""

    # Use combined results if available
    results = state.combined_results if state.combined_results else []

    # If we don't have combined results, try individual result types
    if not results:
        if state.faiss_results:
            results.extend(state.faiss_results)
        if state.bm25_results:
            results.extend(state.bm25_results)
        if state.tavily_results:
            results.extend(state.tavily_results)

    # Format each result with the query that produced it (if available)
    for i, result in enumerate(results):
        results_text += f"RESULT {i+1}:\n"
        results_text += f"Title: {result.title}\n"
        results_text += f"URL: {result.url}\n"
        if result.query:
            results_text += f"Query: {result.query}\n"
        results_text += f"Content: {result.content}\n"
        if result.score is not None:
            results_text += f"Relevance Score: {result.score:.4f}\n"
        results_text += "\n"

    return results_text

def format_search_details(state: SearchState) -> str:
    """Format the search details for the answer generation."""
    details = f"Total search iterations: {state.current_iteration}\n\n"
    details += f"Queries used:\n"

    # Add the original query
    details += f"- Original query: {state.original_query}\n"

    # Add the refined query if present
    if state.refined_query:
        details += f"- Refined query: {state.refined_query}\n"

    # Add all the generated queries
    for i, query in enumerate(state.generated_queries):
        if query != state.refined_query and query != state.original_query:
            details += f"- {query}\n"

    # Add knowledge gaps that were identified
    if state.knowledge_gaps:
        details += f"\nKnowledge gaps identified during search:\n"
        for gap in state.knowledge_gaps:
            details += f"- {gap}\n"

    return details

def deep_reasoning_agent(state: SearchState, max_iterations: int = 3) -> SearchState:
    """
    Uses deep reasoning to analyze results, identify knowledge gaps, and decide if further search is needed.

    Args:
        state: The current search state with combined search results
        max_iterations: Maximum number of search iterations allowed

    Returns:
        Updated state with analysis and potentially new search queries
    """
    # Check if we have any search results to work with
    if (not state.combined_results and
        not state.faiss_results and
        not state.bm25_results and
        not state.tavily_results):
        # No results, set search as complete with appropriate message
        state.search_complete = True
        state.key_points = ["No relevant information found for the query."]
        state.final_answer = "I couldn't find relevant information to answer your query."
        state.confidence_score = 0.1
        return state

    # Initialize the LLM
    llm = init_reasoning_llm()

    # Create the reasoning prompt
    reasoning_prompt = PromptTemplate(
        input_variables=["original_query", "iteration", "search_results", "max_iterations"],
        template=REASONING_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = reasoning_prompt | llm

    # Format the search results
    formatted_results = format_search_results(state)

    # Generate the analysis and reasoning
    response = chain.invoke({
        "original_query": state.original_query,
        "iteration": state.current_iteration,
        "search_results": formatted_results,
        "max_iterations": max_iterations
    })

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        analysis_text = response.content
    else:
        analysis_text = response

    # Parse the JSON response
    try:
        analysis = json.loads(analysis_text)

        # Update the state with the analysis results
        state.key_points = analysis.get("key_points", [])
        state.knowledge_gaps = analysis.get("knowledge_gaps", [])
        state.search_complete = analysis.get("search_complete", False)

        # If we need to continue searching, add new queries
        if not state.search_complete and "new_queries" in analysis and analysis["new_queries"]:
            state.generated_queries = analysis["new_queries"]
            logger.info(f"Generated {len(state.generated_queries)} new queries based on knowledge gaps")
            for i, query in enumerate(state.generated_queries):
                logger.info(f"  New query {i+1}: {query}")

        # Log the reasoning
        if "reasoning" in analysis:
            logger.info(f"Reasoning: {analysis['reasoning']}")

    except json.JSONDecodeError:
        logger.error("Failed to parse deep reasoning JSON output")
        # Try to extract JSON-like content using regex
        import re
        try:
            # Look for JSON-like pattern in the text
            json_match = re.search(r'(\{.*\})', analysis_text, re.DOTALL)
            if json_match:
                json_string = json_match.group(1)
                # Try to parse it again
                analysis = json.loads(json_string)

                # Update the state with the analysis results
                state.key_points = analysis.get("key_points", [])
                state.knowledge_gaps = analysis.get("knowledge_gaps", [])
                state.search_complete = analysis.get("search_complete", False)

                # If we need to continue searching, add new queries
                if not state.search_complete and "new_queries" in analysis and analysis["new_queries"]:
                    state.generated_queries = analysis["new_queries"]
                    logger.info(f"Generated {len(state.generated_queries)} new queries based on knowledge gaps")
                    for i, query in enumerate(state.generated_queries):
                        logger.info(f"  New query {i+1}: {query}")

                logger.info("Successfully extracted JSON using regex")
            else:
                # If we couldn't find JSON, try to extract key points manually
                lines = analysis_text.split('\n')
                key_points = []
                for line in lines:
                    line = line.strip()
                    # Look for bullet points or numbered lists that might be key points
                    if re.match(r'^[\*\-\d\.]\s+', line) and len(line) > 5:
                        # Remove the bullet or number
                        point = re.sub(r'^[\*\-\d\.]+\s+', '', line)
                        key_points.append(point)

                if key_points:
                    state.key_points = key_points
                    logger.info(f"Extracted {len(key_points)} key points manually")
                else:
                    state.key_points = ["Error in analyzing search results."]

                # In case of parsing error, assume search is complete to avoid infinite loops
                state.search_complete = True
        except Exception as e:
            logger.error(f"Failed to extract JSON data: {str(e)}")
            state.search_complete = True
            state.key_points = ["Error in analyzing search results."]

    # If we've reached the maximum iterations, force completion
    if state.current_iteration >= max_iterations:
        state.search_complete = True
        logger.info(f"Reached maximum iterations ({max_iterations}), forcing search completion")

    # Increment the iteration counter
    state.current_iteration += 1

    # If search is complete, generate the final answer
    if state.search_complete:
        state = generate_final_answer(state)

    return state

def generate_final_answer(state: SearchState) -> SearchState:
    """
    Generates the final, structured answer when search is complete.

    Args:
        state: The current search state with key points and other information

    Returns:
        Updated state with the final structured answer
    """
    # Initialize the LLM with a more creative temperature
    llm = init_reasoning_llm(temperature=0.4)

    # Create the answer generation prompt
    answer_prompt = PromptTemplate(
        input_variables=["original_query", "key_points", "search_details"],
        template=ANSWER_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = answer_prompt | llm

    # Format the key points
    key_points_text = "\n".join([f"- {point}" for point in state.key_points])

    # Format the search details
    search_details = format_search_details(state)

    # Generate the final answer
    response = chain.invoke({
        "original_query": state.original_query,
        "key_points": key_points_text,
        "search_details": search_details
    })

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = response

    # Update the state with the final answer
    state.final_answer = answer.strip()

    # Set a reasonable confidence score - could be improved with more advanced heuristics
    state.confidence_score = 0.8 if state.key_points else 0.5

    return state