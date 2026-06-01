# ⚙️ Multiagent Workflow Simulator

**A LangGraph-powered multi-agent pipeline that orchestrates retrieval, reasoning, risk analysis, and synthesis for international development queries.**

---

## Why I Built This

The Sector AI Agent I built earlier is a single RAG pipeline — one model doing everything. But real AI systems in production often need specialized agents working together: one to find information, one to reason about it, one to assess risks, one to write the final response.

This project demonstrates that architecture — a coordinated pipeline of five specialized agents, each with a distinct role, passing state between them using LangGraph's graph-based orchestration.

---

## The Agent Pipeline

```
User Query
     ↓
① Router Agent      — detects domain (Transport/WASH/Agriculture/FCS/Education)
     ↓
② Retrieval Agent   — fetches relevant sector documents from knowledge base
     ↓
③ Reasoning Agent   — analyzes documents, extracts structured insights
     ↓
④ Risk Agent        — identifies operational risks with severity ratings
     ↓
⑤ Synthesis Agent   — combines all outputs into a final actionable response
```

Each agent has its own system prompt, its own role, and adds to a shared state object that flows through the graph. LangGraph manages the execution order and state transitions.

---

## Live Demo

🌐 **Simulator UI:** https://peps143.github.io/multiagent-simulator/frontend/index.html

⚙️ **Backend API:** https://multiagent-simulator.onrender.com

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 0.2.56 |
| LLM | OpenAI GPT-4o-mini |
| Agent framework | LangChain 0.3.25 |
| API backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Backend hosting | Render (free tier) |
| Frontend hosting | GitHub Pages |

---

## Run Locally

```bash
git clone https://github.com/peps143/multiagent-simulator.git
cd multiagent-simulator/backend

pip install -r requirements.txt

# Set your OpenAI key
set OPENAI_API_KEY=sk-your-key-here   # Windows
export OPENAI_API_KEY=sk-your-key-here # Mac/Linux

uvicorn server:app --reload --port 8001
```

Then open `frontend/index.html` in your browser.

---

## Project Structure

```
multiagent-simulator/
├── backend/
│   ├── agents.py        # LangGraph pipeline + all 5 agents
│   ├── server.py        # FastAPI REST server
│   └── requirements.txt
├── frontend/
│   └── index.html       # Simulator UI with live pipeline visualizer
└── README.md
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/run` | Run the full 5-agent pipeline |
| `GET` | `/health` | Health check |

---

## What I Learned

Building this made me think carefully about *state design* — what information each agent needs, what it produces, and how to pass that cleanly between steps. LangGraph's StateGraph forces you to be explicit about this in a way that makes the architecture very clear.

The risk agent was the most interesting to build. Getting it to return structured JSON reliably required careful prompt engineering — including a regex fallback for when the model adds extra text around the JSON.

---

## How This Connects to the Sector AI Agent

This project is the natural evolution of the Sector AI Agent:

- **Sector AI Agent** — single RAG pipeline, great for direct Q&A
- **Multiagent Simulator** — orchestrated pipeline, adds reasoning depth and risk awareness

In a real deployment you'd combine them: the Retrieval Agent would call the Sector AI Agent's FAISS index, and the results would flow through the reasoning and risk agents before synthesis.

---

## What's Next

- Wire the Retrieval Agent to the actual Sector AI Agent FAISS vector store
- Add parallel agent execution (Reasoning + Risk running simultaneously)
- Add a human-in-the-loop checkpoint where users can approve before synthesis
- Stream agent outputs token-by-token for real-time UX

---

*Built by Perpetual T. Adu — international development professional exploring AI orchestration and multi-agent systems.*
