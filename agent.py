from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os

from vector_search import search_job_postings

load_dotenv()

# --- Tools ---
job_search_tool = Tool(
    name="search_job_postings",
    description="Search job postings database for technology trends and skills. Use this when asked about popular frameworks, technologies, or job market trends.",
    func=search_job_postings,
    return_direct=True
)

search = GoogleSearchAPIWrapper(k=3)
google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent tech results that might otherwise be missed by job postings",
    func=search.run,
)

# --- Prompt ---
PROMPT_TEMPLATE = [
    ("system",
    """You are a tech job market analyst.
WEB SEARCH OVERRIDE: {web_search}. If "true", CALL ONLY 'google_search' and DO NOT CALL any other tool.

Your task: answer the question by deciding whether to respond directly, use job postings, or use google search.

Decision rules:
- If web_search=true, you must use the google_search tool regardless of the question, ignore all other rules.
- If the question is general and does not need job postings, like "hi" or "who are you", answer directly
- If the question is about latest news that might be missed by job postings, like "latest Python update" or "Langchain version 2.1", you must use the google search tool
- If the question is about technology, like "What is the most popular frontend framework", or "What technology is used alongside Docker", you must use the job search tool
- If the question asks about required skills for a specific tech role (including internships) or how to prepare for a role, e.g., "what skills should I learn to land a ML internship", you must use the job search tool
- If the question is clearly not about tech (e.g., "what's the weather", "tell me a joke"), say "please ask a tech question"

When answering technology questions, always use the appropriate search tool first to gather information. Only invoke the appropriate tool once.

Important: When invoking the 'search_job_postings' tool, pass the user's original question string exactly as received, with no rephrasing, additions, or formatting changes.

Provide your final answer as plain text, with no extra formatting."""),
    ("human", "web_search={web_search}\n{question}"),
    ("placeholder", "{agent_scratchpad}")
]
prompt = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)

# --- Agent ---
tools = [job_search_tool, google_search_tool]
agent = create_tool_calling_agent(
    llm=ChatGoogleGenerativeAI(model=os.getenv("LLM_MODEL"), temperature=0.3),
    prompt=prompt,
    tools=tools,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_query(query, web_search) -> str:
    raw_response = agent_executor.invoke({"question": query, "web_search": "true" if web_search else "false"})
    return raw_response.get("output")
