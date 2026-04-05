from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

@app.post("/research")
def run_research(query: str):
    try:
        from agents.planner import plan
        from agents.researcher import research
        from agents.writer import write

        plan_text = plan(query)
        research_data = research(plan_text)
        final_text = write(research_data)

        return {
            "success": True,
            "plan": plan_text,
            "research": research_data,
            "final": final_text
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }