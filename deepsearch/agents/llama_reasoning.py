import os
import logging
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from deepsearch.models import SearchState

# Set up logging
logger = logging.getLogger("deepsearch.reasoning")


# Get the OpenAI-compatible API base URL and API key
openai_api_base = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
openai_api_key = os.environ.get("LLM_API_KEY", "not-needed")

# Define the prompt template for reasoning
REASONING_TEMPLATE = """You are a helpful AI assistant tasked with answering questions based on search results.

USER QUERY: {query}

SEARCH RESULTS:
{search_results}

Based on the search results above, provide a comprehensive and accurate answer to the user's query.
Use only information from the search results and avoid making up facts.
If the search results don't contain enough information to answer the query confidently,
acknowledge this limitation and provide the best possible answer with the available information.

Your answer should be well-structured, clear, and directly address the user's query.
Include relevant details from the search results and cite sources when appropriate.

IMPORTANT INSTRUCTIONS:
1. DO NOT include any follow-up questions at the end of your answer.
2. DO NOT ask if the user wants more information or to elaborate further.
3. DO NOT end your response with phrases like "Do you want me to explain more?" or similar.
4. Provide a complete, self-contained answer that stands on its own.
5. Be confident and definitive in your response.
6. Include in-text citations using the format [Author/Article, Year](PMID:$PMID) (for example [Xi et al.](PMID:$PMID)) for each fact or claim. For example, [Smith et al., 2024](PMID:1234567890). The $PMID is the PubMed ID of the article, mentioned in each search result.
7. For each citation, include a brief context about the source (e.g., "A study by [Smith et al., 2024](PMID:1234567890) found that...")

Answer:
"""

def init_reasoning_llm():
    """Initialize the language model for reasoning using OpenAI-compatible API."""
    # Use OpenAI-compatible server
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "local-model"),
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base if not openai_api_key or openai_api_key == "not-needed" else None,
        temperature=0.3,
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
        if state.pubmed_results:
            results.extend(state.pubmed_results)

    # Format each result
    for i, result in enumerate(results):
        results_text += f"RESULT {i+1}:\n"
        results_text += f"Title: {result.title}\n"
        results_text += f"URL: {result.url}\n"
        results_text += f"Content: {result.content}\n"
        if result.score is not None:
            results_text += f"Relevance Score: {result.score:.4f}\n"
        results_text += "\n"

    return results_text

def calculate_confidence(answer: str) -> float:
    """Calculate a confidence score based on the answer text."""
    # This is a simple heuristic - you might want to use a more sophisticated approach
    confidence = 0.7  # Higher base confidence (was 0.5)

    # Check for confidence indicators in the text
    if "not enough information" in answer.lower() or "don't have enough" in answer.lower():
        confidence -= 0.3
    if "according to" in answer.lower() or "based on the search results" in answer.lower():
        confidence += 0.2
    if "uncertain" in answer.lower() or "unclear" in answer.lower():
        confidence -= 0.2
    if "clearly" in answer.lower() or "definitively" in answer.lower():
        confidence += 0.1

    # Cap confidence between 0 and 1
    return max(0.0, min(1.0, confidence))

def llama_reasoning_agent(state: SearchState) -> SearchState:
    """
    Uses Llama.cpp to analyze results and generate a cohesive answer.

    Args:
        state: The current search state with combined search results

    Returns:
        Updated state with the final answer and confidence score
    """
    # Check if we have any search results to work with
    if (not state.combined_results and
        not state.faiss_results and
        not state.bm25_results and
        not state.tavily_results and
        not state.pubmed_results):
        # No results, set a low confidence and an appropriate message
        state.final_answer = "I couldn't find relevant information to answer your query."
        state.confidence_score = 0.1
        return state

    # Initialize the LLM
    llm = init_reasoning_llm()

    # Create the prompt
    reasoning_prompt = PromptTemplate(
        input_variables=["query", "search_results"],
        template=REASONING_TEMPLATE
    )

    # Use the newer approach to avoid deprecation warnings
    chain = reasoning_prompt | llm

    # Format the search results
    formatted_results = format_search_results(state)

    # Use refined query if available, otherwise use original query
    query = state.original_query

    # Generate the answer
    response = chain.invoke({
        "query": query,
        "search_results": formatted_results
    })

    # Extract the content if it's a message object
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = response

    # Calculate confidence score
    confidence = calculate_confidence(answer)

    # Update the state
    state.final_answer = answer.strip()
    state.confidence_score = confidence

    return state