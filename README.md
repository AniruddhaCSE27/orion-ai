# ORION AI

ORION AI is a production-ready Autonomous Research and Web-based RAG system built with FastAPI, Streamlit, OpenAI, Tavily, and FAISS.

It keeps the premium answer-first UI while using a deterministic backend pipeline:

`Query -> Classification -> Tavily Retrieval -> Evidence Normalization -> Lightweight Reranking -> Writer -> Grounded Answer -> Report/PDF`

## What ORION Does

- Runs web-grounded research using Tavily
- Normalizes and reranks evidence before generation
- Produces answer-first responses instead of generic fallback text
- Preserves conversational memory for short follow-up context
- Exports the final answer and sources as PDF
- Exposes a FastAPI backend plus a Streamlit frontend

## Core Stack

- Python
- FastAPI
- Streamlit
- OpenAI API
- Tavily API
- FAISS (`faiss-cpu`)
- ReportLab
- `python-dotenv`

## Active Entry Points

Use these entry points for local runs and deployment:

- Streamlit frontend: `streamlit_app.py`
- Backend API: `backend.main:app`

Legacy root files now forward to the maintained frontend/backend modules so local runs and deployments stay aligned.

## Environment Variables

Create a `.env` file from `.env.example`.

Required:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Recommended:

```env
BACKEND_URL=http://localhost:8000
MODEL_NAME=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_TIMEOUT_SECONDS=30
```

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the backend:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

3. In a second terminal, start the frontend:

```bash
streamlit run streamlit_app.py
```

4. Open the Streamlit app in your browser.

## Deployment

### Streamlit Community Cloud

1. Deploy the repo and set the app entry file to `streamlit_app.py`.
2. Add these secrets in the Streamlit dashboard:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `BACKEND_URL`
   - optional: `MODEL_NAME`, `EMBEDDING_MODEL`, `OPENAI_TIMEOUT_SECONDS`
3. Point `BACKEND_URL` to your deployed FastAPI backend.

### Render or Similar Backend Hosting

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Set the same environment variables on the backend service, except `BACKEND_URL` is only needed by the frontend deployment.

## Response Contract

The backend returns a stable JSON structure shaped like:

```json
{
  "success": true,
  "query": "best ai tools for students",
  "mode": "Web Recommendations",
  "answer": "...",
  "direct_answer": "...",
  "plan": "...",
  "evidence": [],
  "sources": [],
  "source_count": 4,
  "report": "...",
  "report_word_count": 120,
  "plan_word_count": 35,
  "debug": {
    "raw_result_count": 4,
    "usable_evidence_count": 4,
    "reranked_evidence_count": 4,
    "writer_output_length": 220,
    "fallback_triggered_reason": ""
  },
  "error": ""
}
```

If no usable search results exist, ORION returns `No relevant live data found.` and an empty source list.

## Verification

Recommended checks before shipping:

```bash
python -m py_compile backend/main.py backend/api/routes.py backend/services/research_service.py backend/agents/writer.py frontend/app.py app.py main.py streamlit_app.py
```

## Notes

- The frontend design remains intact while the answer rendering path now prefers real backend output.
- The backend no longer exposes the broken document indexing path.
- FAISS reranking/cache is best-effort; if it fails, ORION still answers using normalized Tavily evidence.
