# Job Market AI - Backend

An AI agent that answers tech job market questions and recommends skills to learn based on your resume and a corpus of real job postings.

It combines an agentic LLM with hybrid retrieval (dense + BM25) over a PGVector database of job postings, plus an optional live web search fallback for freshest results.

> **Demo:**  
> See the animation below for a demonstration of the Job Market AI.

![Demo of Job Market AI](demo.gif)

## Features

- **Tech job market Q&A**: Ask about trends, popular frameworks, and technology stacks.
- **Resume-driven skill gaps**: Upload a PDF resume; get 1-3 high-impact technical skills that synergize with your background.
- **Hybrid retrieval**: Ensemble of dense vector search and BM25 keyword search, with multi-query expansion via LLM.
- **Agent tools**: Vector DB search tool for postings and Google search tool for recent news.
- **FastAPI backend + React (Vite) frontend** with a clean chat UI and a web-search toggle.

## Architecture

```mermaid
graph TD
  U["User"] --> FE["React + Vite Frontend"]
  FE -->|"/query, /upload_resume"| BE["FastAPI Backend"]
  BE --> AG["LangChain Agent"]
  AG -->|"Tool: search_job_postings"| VS["PGVector (Postgres + pgvector)"]
  AG -->|"Tool: google_search"| GS["Google Custom Search"]
  VS --> EMB["Google Generative AI Embeddings"]
  BE --> GEM["Gemini (Google GenAI)"]
```

## Tech Stack

- **Frontend**: React, Vite, Tailwind CSS
- **Backend**: FastAPI, Pydantic
- **LLM**: Gemini via `langchain-google-genai`
- **Retrieval**: `langchain-postgres` (PGVector), `BM25Retriever`, `EnsembleRetriever`, `MultiQueryRetriever`
- **Search**: Google Custom Search via `langchain_google_community.GoogleSearchAPIWrapper`

## Prerequisites

- Python 3.10+
- Docker (for Postgres + pgvector) or an existing Postgres with the `pgvector` extension
- Google API keys:
  - Gemini: `GOOGLE_API_KEY`
  - Google Custom Search: `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` (Programmable Search Engine)

## Environment Variables

Create a `.env` at the project root with:

```
# LLM
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your_google_api_key

# Google Custom Search (for web tool)
GOOGLE_CSE_ID=your_cse_id

# Postgres + PGVector connection string (SQLAlchemy/psycopg)
CONNECTION_STRING=postgresql+psycopg://postgres:postgres@localhost:5432/job_market_ai
```

Notes:

- `GoogleSearchAPIWrapper` uses `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`.
- Ensure the `pgvector` extension is installed in your Postgres database.

## Start Postgres with pgvector (Docker)

```
docker run --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=job_market_ai \
  -p 5432:5432 -d ankane/pgvector:latest

# Optionally enable extension explicitly if needed
docker exec -it pgvector psql -U postgres -d job_market_ai -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Bootstrap the Vector Database

1. Place your PDFs into subfolders under `data/`:
   - `data/AI/`
   - `data/Full stack/`
   - `data/Software/`
2. From the project root, run:

```
pip install -r backend/requirements.txt
python scripts/setup_vectordb.py
```

This script:

- Loads PDFs and splits them into overlapping chunks.
- Creates embeddings via Gemini (`models/embedding-001`).
- Writes chunks and vectors into the `job_chunks` collection in PGVector.

## Run the Backend (FastAPI)

From the project root:

```
pip install -r backend/requirements.txt
python backend/main.py
# or: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## Deploying on AWS

- Backend: Deploy FastAPI on EC2, ECS/Fargate, or Elastic Beanstalk. Expose port 8000 (or behind a load balancer) and set environment variables (`LLM_MODEL`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `CONNECTION_STRING`).
- Database: Use Amazon RDS for Postgres with the `pgvector` extension (preferred) or run `ankane/pgvector` on ECS/EC2. Update `CONNECTION_STRING` accordingly and rerun the vector DB bootstrap with your PDFs.
- Networking: Restrict security groups to required ports, and set proper HTTPS termination on ALB/CloudFront.

### Endpoints

- `POST /query`
  - Body: `{ "query": string, "web_search": boolean }`
  - Response: `{ "response": string }` (empty string if no results)

- `POST /upload_resume`
  - Form-data: `file` (PDF)
  - Response: `{ "response": string }`

### cURL Examples

```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the most popular front end framework?","web_search":false}'

curl -X POST http://localhost:8000/upload_resume \
  -F "file=@/path/to/Resume.pdf"
```

## How It Works

- The agent (`backend/agent.py`) has two tools:
  - `search_job_postings`: Hybrid retrieval over PGVector using an ensemble of dense + BM25, expanded by an LLM via `MultiQueryRetriever`.
  - `google_search`: Google Custom Search for recent news. If the Web Search toggle is on, this tool is used exclusively.
- For resume flow, the app:
  1. Extracts text from the PDF and filters for skills via LLM.
  2. Queries related postings (weighted ensemble) to provide evidence.
  3. Generates 1-3 skill recommendations with rationale.

## Development Notes

- CORS is enabled for `http://localhost:3000` and `http://localhost:5173` in `backend/main.py`.
- The BM25 retriever and DB connections are cached to reduce per-request overhead.
- Expect LLM rate limits; the UI surfaces a friendly message on 429/500s.

## Repository Layout

```
Job Market AI/
  backend/               # FastAPI app, agent, retrieval, skills recommender
  frontend/              # React + Vite client
  scripts/               # One-off scripts (vector DB setup)
  data/                  # Put PDFs here (AI / Full stack / Software)
  uploads/               # Resume uploads at runtime
```

Key files:

- `backend/main.py`: FastAPI app and endpoints.
- `backend/agent.py`: LangChain agent with tools (vector search + Google search).
- `backend/vector_search.py`: Hybrid retriever (dense + BM25), multi-query, LLM synthesis.
- `backend/skills_recommender.py`: Reads resume PDF, extracts skills, recommends complementary skills using postings.
- `scripts/setup_vectordb.py`: Loads PDFs, chunks text, creates embeddings, and writes to PGVector.


