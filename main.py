from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Backend is running 🚀"}

@app.get("/research")
def run_research(query: str):
    try:
        from agents.planner import plan
        from agents.researcher import research
        from agents.writer import write

        plan_data = plan(query)
        research_data = research(plan_data)
        result = write(research_data)

        return {"result": result}

    except Exception as e:
        return {"error": str(e)}