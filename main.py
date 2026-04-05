from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.planner import plan
from agents.researcher import research
from agents.writer import write

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
    return {"message": "ORION AI Backend Running 🚀"}

@app.post("/research")
def run_research(query: str):
    try:
        if not query or not query.strip():
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Query cannot be empty."}
            )

        plan_text = plan(query)
        research_data = research(plan_text)
        final_report = write(plan_text, research_data)

        return {
            "success": True,
            "plan": plan_text,
            "research": research_data,
            "final": final_report
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )