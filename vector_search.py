from dotenv import load_dotenv
from time import sleep
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os

from utils import NoResultsException

load_dotenv()

# Cache for BM25 retriever and database connection to avoid rebuilding on every query
_bm25_retriever = None
_db_instance = None
_embeddings_instance = None

# Prompt template for LLM queries
PROMPT_TEMPLATE = """
You are a software job market analyst. Answer the question based on the job posting data provided below.

Context from software job postings:
{context}

Question: {question}

Instructions:
- Try to answer the question based on available context, even if the context is not directly relevant or sufficient
- Provide specific examples from the context when possible
- If asked about "most popular" or trends, analyze the frequency of mentions across the job postings
"""

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return _embeddings_instance

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = PGVector(
            collection_name="job_chunks",
            connection=os.getenv("CONNECTION_STRING"),
            embeddings=get_embeddings()
        )
    return _db_instance

def get_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        db = get_db()  # Use cached database instance
        # Get all documents for BM25 corpus (only done once)
        all_docs = db.similarity_search("", k=1000)
        _bm25_retriever = BM25Retriever.from_documents(all_docs)
        _bm25_retriever.k = 10
    return _bm25_retriever

# fetch relevant chunks with multiple queries (llm call for multi-query) -> invoke llm with context and prompt (llm call)

def main():
    query = "what is the most popular front end framework"
    response = search_job_postings(query)
    print("\n---\n" + response)

# if results found, response is returned
# if not, exception is raised
def search_job_postings(query) -> str:
    results = get_results(query)
    response_text = invoke_llm(query, results)
    return response_text

def get_results(query, weights=[0.5, 0.5]):
    db = get_db()  # Use cached database instance
    llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0.3)

    # Create retrievers
    dense = db.as_retriever(search_kwargs={"k": 10})
    bm25 = get_bm25_retriever()  # Use cached BM25 retriever

    # Combine dense + BM25
    ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=weights)
    
    # Wrap with MultiQueryRetriever
    mqr = MultiQueryRetriever.from_llm(
        retriever=ensemble, 
        llm=llm,
        include_original=True
    )

    # Add small delay to avoid rate limits
    sleep(0.1)
    results = mqr.invoke(query)
    print(f"Found {len(results)} results")
    if len(results) == 0:
        raise NoResultsException
    return results

def invoke_llm(query, results):
    llm = ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0.3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    if __name__ == "__main__":
        print(context_text)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    return llm.invoke(prompt).content

def get_formatted_response(response, sources):
    return f"Response: {response}\nSources: {sources}"

def get_sources(results):
    # sources = [doc.metadata.get("source", None) for doc in results]
    # avoid duplicates
    sources = []
    seen = set()
    for doc in results:
        source = doc.metadata.get("source", None)
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    return sources

if __name__ == "__main__":
    main()
