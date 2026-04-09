# ORION AI

An autonomous multi-agent RAG research system for turning a simple query into a structured, source-grounded research brief.

ORION AI combines planning, live web research, retrieval, memory, and report generation in a polished Streamlit experience. It is designed to feel like a compact research copilot: take a user prompt, gather evidence from the web, ground the answer with retrieved context, and return a clean output with a research plan, key findings, final report, sources, and PDF export.

## Project Overview

ORION AI is a recruiter-friendly showcase of applied AI engineering across frontend, backend, retrieval, and orchestration layers.

It includes:
- Multi-agent flow for research orchestration
- Real-time web search with Tavily
- FAISS-backed retrieval for source-grounded context
- Lightweight conversational memory for continuity
- Premium Streamlit UI for interactive research workflows
- Professional PDF export for shareable reports

The system is built to produce source-grounded responses designed to reduce hallucinations, while keeping the architecture modular and easy to extend.

## Features

- Planner -> Researcher -> Writer multi-agent pipeline
- RAG pipeline with FAISS-based retrieval over research findings
- Real-time web research using Tavily API
- Source-grounded responses with visible sources and citation-style linking
- Conversational memory for short-term context continuity
- Structured output format:
  - Research Plan
  - Key Findings
  - Final Report
  - Sources
- PDF export with formatted sections, metadata, and source list
- Modular Python project structure with separated frontend, API, services, agents, and core config

## Tech Stack

- Python
- Streamlit
- FastAPI
- OpenAI API
- Tavily API
- FAISS
- ReportLab
- JSON-based lightweight memory store

## Architecture

ORION AI follows a simple but production-style research pipeline:

`Query -> Planner -> Researcher -> Retrieval -> Writer -> Output`

### Flow

1. The user submits a research query from the Streamlit frontend.
2. The Planner creates a focused research plan.
3. The Researcher performs live web search using Tavily.
4. Research findings are embedded and stored in a FAISS vector store.
5. Relevant `retrieved_context` is pulled back from the vector store.
6. The Writer uses:
   - the research plan
   - retrieved context
   - live findings
   - recent conversation context
7. The system returns a structured response:
   - Research Plan
   - Key Findings
   - Final Report
   - Sources
8. The frontend renders the output and allows PDF export.

### Why Retrieved Context Matters

The `retrieved_context` step grounds the final writing stage in the most relevant evidence collected during research. Instead of relying only on the LLM's general knowledge, ORION AI passes retrieved source snippets into the writer so the final answer stays closer to gathered evidence.

## Installation

### Clone the repository

```bash
git clone https://github.com/AniruddhaCSE27/orion-ai.git
cd orion-ai
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run frontend/app.py
```

If you want to run the API layer independently, the backend app entry point is:

```bash
uvicorn backend.main:app --reload
```

## Environment Variables

Create a `.env` file in the project root.

Required variables:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Optional configuration:

```env
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

An example file is already included:

`/.env.example`

## Project Structure

```text
orion-ai/
├── frontend/
│   ├── app.py
│   └── utils/
│       ├── __init__.py
│       └── charts.py
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── analyst.py
│   │   ├── critic.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   └── writer.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data/
│   │   └── memory.json
│   ├── models/
│   │   └── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── research_service.py
│   │   └── vector_store.py
│   ├── __init__.py
│   └── main.py
├── agents/                 # legacy compatibility copies
├── tools/                  # legacy compatibility copies
├── utils/                  # legacy compatibility copies
├── .env.example
├── app.py                  # legacy compatibility entry
├── main.py                 # legacy compatibility entry
├── memory.db
├── requirements.txt
└── README.md
```

## Live Demo

- Streamlit frontend: Add your deployed Streamlit URL here once published
- Live backend API: [ORION Backend on Render](https://orion-backend-s0e6.onrender.com)

If a public Streamlit deployment already exists outside this repository, replace the placeholder above with the actual app URL.

## Screenshots

Add screenshots here after deployment polish:

- `docs/screenshots/home.png`
- `docs/screenshots/report-view.png`
- `docs/screenshots/pdf-export.png`

Suggested markdown:

```md
![Home UI](docs/screenshots/home.png)
![Report View](docs/screenshots/report-view.png)
![PDF Export](docs/screenshots/pdf-export.png)
```

## Future Improvements

- Document upload and PDF chat
- Evaluation metrics for retrieval quality and answer grounding
- Pinecone integration for scalable vector storage
- User authentication and saved workspaces
- Background jobs for longer research tasks

## Why This Project Stands Out

ORION AI demonstrates more than prompt chaining. It shows end-to-end applied AI product thinking:

- modular backend architecture
- retrieval-augmented generation
- source-grounded answer design
- conversational context management
- UI/UX polish for AI workflows
- export-ready reporting

This makes it a strong portfolio project for roles involving AI engineering, LLM apps, full-stack product development, and agentic workflows.

## Author

**Aniruddha Pathak**

## License

Add a license file if you want to open-source the project formally.
