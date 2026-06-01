"""
FastAPI server for the Multiagent Workflow Simulator
Exposes the LangGraph pipeline as a REST API
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import run_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Multiagent Simulator API ready")
    yield

app = FastAPI(
    title="Multiagent Workflow Simulator",
    description="LangGraph-powered multi-agent pipeline for international development queries",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/run")
def run_agents(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    try:
        result = run_pipeline(req.query)
        return {
            "query":          result["query"],
            "domain":         result["domain"],
            "retrieved_docs": result["retrieved_docs"],
            "reasoning":      result["reasoning"],
            "risks":          result["risks"],
            "final_answer":   result["final_answer"],
            "agent_trace":    result["agent_trace"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
    }

@app.get("/")
def root():
    return {"message": "Multiagent Workflow Simulator API — visit /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
