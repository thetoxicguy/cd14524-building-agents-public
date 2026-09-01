#!/usr/bin/env python3
"""
UdaPlay Agent Implementation
Runs the gaming AI analytics agent with retrieval, evaluation, and web search tools
"""

import os
import sys
import json
import importlib.util
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient
from pydantic import BaseModel

# Handle pysqlite3 for Udacity
if importlib.util.find_spec("pysqlite3") is not None:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment
load_dotenv(find_dotenv(), override=True)

# Import from lib
sys.path.insert(0, '/Users/daniel.a.robles/development/cd14524-building-agents-public')
from lib.agents import Agent
from lib.llm import LLM
from lib.tooling import tool

# Verify API keys
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
assert os.getenv("OPENAI_BASE_URL"), "OPENAI_BASE_URL is not set"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize LLM client
client = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Load ChromaDB collection from Part 1
print("Loading ChromaDB collection...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
collection = chroma_client.get_or_create_collection(
    name="games",
    embedding_function=embedding_fn
)
print(f"✓ ChromaDB collection loaded: {collection.count()} games available")

# Define Tools

class EvaluationReport(BaseModel):
    """Evaluation report for retrieved documents"""
    is_relevant: bool
    confidence: float
    reasoning: str

@tool(name="retrieve_game", description="Semantic search: Finds game results in the vector DB using ChromaDB. Returns the most relevant games matching the query with platform, name, year, and confidence score.")
def retrieve_game(query: str) -> dict:
    """
    Retrieves games from ChromaDB based on semantic similarity to the query.
    """
    results = collection.query(query_texts=[query], n_results=3)
    
    retrieved_games = []
    if results['documents'] and len(results['documents']) > 0:
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i] if results['distances'] else 0
            confidence = 1 - distance
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            
            retrieved_games.append({
                'document': doc,
                'platform': metadata.get('platform', 'Unknown'),
                'name': metadata.get('name', 'Unknown'),
                'year': metadata.get('year', 'Unknown'),
                'confidence': round(confidence, 2)
            })
    
    return {
        'results': retrieved_games,
        'query': query,
        'num_results': len(retrieved_games)
    }

@tool(name="evaluate_retrieval", description="Evaluates if retrieved documents are relevant and sufficient to answer the user's question.")
def evaluate_retrieval(query: str, retrieval_result: dict) -> dict:
    """
    Uses an LLM to evaluate if retrieved documents are sufficient to answer the query.
    """
    documents_text = "\n".join([
        f"- {r['name']} ({r['platform']}, {r['year']}): {r['document']}"
        for r in retrieval_result.get('results', [])
    ])
    
    eval_prompt = f"""Your task is to evaluate if the retrieved documents are sufficient to answer the user's question.

User Question: {query}

Retrieved Documents:
{documents_text}

Provide a JSON response with:
- "is_relevant": boolean
- "confidence": number 0.0-1.0
- "reasoning": string explanation"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.3
    )
    
    eval_text = response.choices[0].message.content
    
    try:
        json_start = eval_text.find('{')
        json_end = eval_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = eval_text[json_start:json_end]
            eval_data = json.loads(json_str)
            return {
                'is_relevant': eval_data.get('is_relevant', True),
                'confidence': float(eval_data.get('confidence', 0.5)),
                'reasoning': eval_data.get('reasoning', eval_text)
            }
    except Exception as e:
        pass
    
    return {
        'is_relevant': True,
        'confidence': 0.6,
        'reasoning': f"Evaluation: {eval_text}"
    }

@tool(name="game_web_search", description="Performs web search using Tavily API to find current information about games.")
def game_web_search(query: str) -> dict:
    """
    Searches the web for game-related information using Tavily API.
    """
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        search_response = tavily_client.search(query, max_results=5)
        
        answer = search_response.get("answer", "No answer found")
        sources = [result.get("url", "") for result in search_response.get("results", [])]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 0.8,
            'query': query
        }
    except Exception as e:
        return {
            'answer': f"Web search failed: {str(e)}",
            'sources': [],
            'confidence': 0.0,
            'query': query,
            'error': str(e)
        }

# Create Agent
print("\nCreating Agent...")

system_prompt = """You are an expert gaming AI assistant specializing in video game information and history.

Your primary role is to:
1. Answer questions about video games using the game database (retrieve_game tool)
2. Evaluate if retrieved information is sufficient (evaluate_retrieval tool)
3. Fall back to web search for current or missing information (game_web_search tool)

Guidelines:
- First, try to retrieve information from the internal game database
- Evaluate the relevance of retrieved results
- If results are not relevant (confidence < 0.5) or insufficient, use web search
- Always provide sources and confidence levels in your responses
- Be accurate and cite game platforms, years, and publishers when available
- If you're unsure about information, acknowledge the uncertainty"""

agent = Agent(
    model="gpt-4o-mini",
    instructions=system_prompt,
    tools=[retrieve_game, evaluate_retrieval, game_web_search],
    temperature=0.3
)

print("✓ Agent created successfully with 3 tools")

# Test queries
test_queries = [
    "When was Pokémon Gold and Silver released?",
    "Which one was the first 3D platformer Mario game?",
    "Was Mortal Kombat X released for Playstation 5?"
]

print("\n" + "=" * 70)
print("AGENT TEST QUERIES")
print("=" * 70)

for i, query in enumerate(test_queries, 1):
    print(f"\n[Query {i}] {query}")
    print("-" * 70)
    
    try:
        response = agent.invoke(query, session_id=f"test_session_{i}")
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("✓ Agent testing complete")
print("=" * 70)
