"""
Multiagent Workflow Simulator
5 specialized agents orchestrated by LangGraph for international development queries.

Agent pipeline:
  Router → Retrieval → Reasoning → Risk → Synthesis
"""

import os
from typing import TypedDict, Annotated, Literal
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── Shared state passed between all agents ─────────────────────────────────────
class AgentState(TypedDict):
    query:          str                  # original user question
    domain:         str                  # detected sector domain
    retrieved_docs: list[dict]           # docs from retrieval agent
    reasoning:      str                  # analysis from reasoning agent
    risks:          list[dict]           # risks from risk agent
    final_answer:   str                  # synthesized final response
    agent_trace:    list[dict]           # step-by-step trace of all agents
    current_agent:  str                  # which agent is running now
    error:          str                  # any error message


# ── Knowledge base (simulated sector documents) ────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": "KB-001", "sector": "Transport",
        "title": "Rural Roads ICR — P123456",
        "content": """Community ownership is critical for road maintenance. Projects that 
        established local road committees achieved 40% better maintenance outcomes over 5 years.
        Climate resilience design was underweighted — 3 of 7 road segments suffered damage in 
        the second rainy season. Procurement delays averaged 8.3 months due to single-source 
        bidding in remote areas. Gender-disaggregated beneficiary data was not collected until 
        Year 3."""
    },
    {
        "id": "KB-002", "sector": "Agriculture",
        "title": "Digital Agriculture Sector Note — ITSEF",
        "content": """Last-mile connectivity remains the binding constraint. Of 12 projects, 
        9 reported that digital tools failed to reach the bottom 40% of farmers. SMS-based 
        systems outperformed app-based in low-literacy contexts (85th vs 34th percentile adoption).
        Data sovereignty concerns unresolved in 7 of 12 project legal frameworks. Only 2 of 12 
        projects had government-funded continuation plans."""
    },
    {
        "id": "KB-003", "sector": "WASH",
        "title": "Urban WASH PAD — P198765",
        "content": """Tariff reform sequencing is critical — quality improvements must precede 
        cost recovery. Non-Revenue Water baseline of 52% traced to illegal connections (60%) 
        and aging pipes (40%). Women in governance: mandatory 30% female representation on 
        Water User Committees increased complaint resolution by 55%. Cross-ministerial 
        coordination between MoF and MoW causes 6-month delays."""
    },
    {
        "id": "KB-004", "sector": "Education",
        "title": "Education ISR — West Africa",
        "content": """2,300 trained teachers remain undeployed due to Ministry of Finance 
        hiring freeze — project design did not include fiscal space analysis. 47% of schools 
        built in flood-prone areas despite climate screening requirements. Reading scores 
        improved by 0.3 SDs in Grades 1-3 but showed no improvement in Grade 4-6. 
        Capacity substitution delays institutionalization."""
    },
    {
        "id": "KB-005", "sector": "FCS",
        "title": "FCS Portfolio Synthesis — ITSEF",
        "content": """Projects with adaptive design mechanisms achieved 71% of PDO targets 
        vs 43% for traditionally designed projects. First disbursements within 90 days of 
        effectiveness increased community participation by 34%. Remote monitoring triangulation 
        (satellite + community scorecards + third-party) is most reliable. Heavy reliance on 
        international NGOs hollowed out government capacity in 12 projects."""
    },
    {
        "id": "KB-006", "sector": "Cross-Cutting",
        "title": "Climate-Smart Agriculture Synthesis — SSA",
        "content": """Climate data infrastructure must be treated as Year 0-1 priority, not 
        a project component. Gender-responsive design beyond quotas: projects using female 
        extension workers saw 2.4x higher adoption among women farmers. Carbon finance 
        transaction costs for smallholders averaged $47/tonne vs $8 for commercial. 
        Agroforestry shows highest long-term potential but weakest within-project results."""
    },
]

DOMAIN_KEYWORDS = {
    "Transport":    ["road","transport","bridge","highway","procurement","maintenance","rural road"],
    "Agriculture":  ["agri","farm","crop","food","irrigation","extension","digital agri","climate-smart","smallholder"],
    "WASH":         ["water","sanitation","wash","tariff","nrw","hygiene","utility","non-revenue"],
    "Education":    ["school","teacher","learning","education","literacy","curriculum","reading"],
    "FCS":          ["fragile","conflict","fcs","post-conflict","humanitarian","displacement","adaptive"],
    "Cross-Cutting":["gender","climate","cross-cutting","carbon","monitoring","capacity"],
}


def detect_domain(query: str) -> str:
    q = query.lower()
    scores = {d: sum(1 for k in kws if k in q) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


def retrieve_docs(query: str, domain: str) -> list[dict]:
    """Simple keyword retrieval from knowledge base."""
    q = query.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        score = 0
        if doc["sector"] == domain or doc["sector"] == "Cross-Cutting":
            score += 3
        words = q.split()
        score += sum(1 for w in words if w in doc["content"].lower() and len(w) > 3)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:3]]


def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key,
    )


def trace_entry(agent: str, input_summary: str, output_summary: str, duration_ms: int) -> dict:
    return {
        "agent":          agent,
        "timestamp":      datetime.now().isoformat(),
        "input_summary":  input_summary,
        "output_summary": output_summary,
        "duration_ms":    duration_ms,
        "status":         "success",
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Router
# Detects domain and decides which agents to prioritize
# ══════════════════════════════════════════════════════════════════════════════
def router_agent(state: AgentState) -> AgentState:
    import time
    t0 = time.time()

    query = state["query"]
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = get_llm(api_key)

    domain = detect_domain(query)

    response = llm.invoke([
        SystemMessage(content="""You are a Router Agent for a World Bank knowledge system.
        Your job is to classify the query and identify the key information needs.
        Be brief — 2-3 sentences max."""),
        HumanMessage(content=f"""Query: {query}
        Detected domain: {domain}
        
        Briefly confirm the domain and state what the retrieval agent should focus on.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Router Agent",
        f"Query: {query[:80]}...",
        f"Domain: {domain} | {response.content[:120]}...",
        duration
    )

    return {
        **state,
        "domain":        domain,
        "current_agent": "retrieval",
        "agent_trace":   state.get("agent_trace", []) + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Retrieval
# Fetches relevant documents from the knowledge base
# ══════════════════════════════════════════════════════════════════════════════
def retrieval_agent(state: AgentState) -> AgentState:
    import time
    t0 = time.time()

    docs = retrieve_docs(state["query"], state["domain"])
    duration = int((time.time() - t0) * 1000)

    trace = trace_entry(
        "Retrieval Agent",
        f"Domain: {state['domain']} | Query: {state['query'][:60]}...",
        f"Retrieved {len(docs)} documents: {', '.join(d['title'] for d in docs)}",
        duration
    )

    return {
        **state,
        "retrieved_docs": docs,
        "current_agent":  "reasoning",
        "agent_trace":    state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Reasoning
# Analyzes retrieved documents and builds structured insights
# ══════════════════════════════════════════════════════════════════════════════
def reasoning_agent(state: AgentState) -> AgentState:
    import time
    t0 = time.time()

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = get_llm(api_key)

    docs_text = "\n\n".join(
        f"[{d['title']}]\n{d['content']}" for d in state["retrieved_docs"]
    )

    response = llm.invoke([
        SystemMessage(content="""You are a Reasoning Agent for a World Bank knowledge system.
        You analyze retrieved documents and extract structured operational insights.
        Focus on: key lessons, patterns across projects, success factors, and evidence.
        Structure your response with clear numbered points."""),
        HumanMessage(content=f"""Query: {state['query']}
        Domain: {state['domain']}
        
        Retrieved Documents:
        {docs_text}
        
        Analyze these documents and extract the most relevant insights for the query.
        Provide 3-5 structured analytical points with evidence from the documents.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Reasoning Agent",
        f"{len(state['retrieved_docs'])} docs analyzed",
        f"Generated {len(response.content.split(chr(10)))} lines of analysis",
        duration
    )

    return {
        **state,
        "reasoning":     response.content,
        "current_agent": "risk",
        "agent_trace":   state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — Risk
# Identifies project risks and flags based on the query context
# ══════════════════════════════════════════════════════════════════════════════
def risk_agent(state: AgentState) -> AgentState:
    import time
    t0 = time.time()

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = get_llm(api_key)

    response = llm.invoke([
        SystemMessage(content="""You are a Risk Agent for a World Bank knowledge system.
        You identify operational risks based on sector knowledge and project experience.
        Return a JSON array of risks with this structure:
        [{"risk": "risk name", "severity": "High/Medium/Low", "mitigation": "brief mitigation"}]
        Return ONLY the JSON array, no other text."""),
        HumanMessage(content=f"""Query: {state['query']}
        Domain: {state['domain']}
        Analysis: {state['reasoning'][:500]}
        
        Identify 3-4 key operational risks relevant to this query.""")
    ])

    # Parse risks safely
    import json, re
    try:
        raw = response.content.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        risks = json.loads(match.group()) if match else []
    except Exception:
        risks = [
            {"risk": "Implementation capacity gap", "severity": "High",   "mitigation": "Early capacity assessment and phased rollout"},
            {"risk": "Stakeholder coordination failure", "severity": "Medium", "mitigation": "Establish inter-agency working group at project start"},
            {"risk": "Sustainability post-completion", "severity": "Medium", "mitigation": "Government ownership and budget integration from Year 1"},
        ]

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Risk Agent",
        f"Analyzed {state['domain']} context",
        f"Identified {len(risks)} risks: {', '.join(r['risk'] for r in risks[:2])}...",
        duration
    )

    return {
        **state,
        "risks":         risks,
        "current_agent": "synthesis",
        "agent_trace":   state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 — Synthesis
# Combines all agent outputs into a final coherent response
# ══════════════════════════════════════════════════════════════════════════════
def synthesis_agent(state: AgentState) -> AgentState:
    import time
    t0 = time.time()

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = get_llm(api_key)

    risks_text = "\n".join(
        f"- [{r['severity']}] {r['risk']}: {r['mitigation']}"
        for r in state["risks"]
    )
    sources = ", ".join(d["title"] for d in state["retrieved_docs"])

    response = llm.invoke([
        SystemMessage(content="""You are a Synthesis Agent for a World Bank ITSEF knowledge system.
        You combine insights from multiple specialized agents into a clear, actionable response.
        Write for a development professional audience. Be specific, cite evidence, and be practical.
        Structure: brief direct answer → key insights → recommendations."""),
        HumanMessage(content=f"""Query: {state['query']}
        Domain: {state['domain']}
        
        Reasoning Agent Analysis:
        {state['reasoning']}
        
        Risk Agent Findings:
        {risks_text}
        
        Sources used: {sources}
        
        Synthesize a final comprehensive response to the query.""")
    ])

    duration = int((time.time() - t0) * 1000)
    trace = trace_entry(
        "Synthesis Agent",
        "Combining reasoning + risk outputs",
        f"Generated final response ({len(response.content)} chars)",
        duration
    )

    return {
        **state,
        "final_answer":  response.content,
        "current_agent": "complete",
        "agent_trace":   state["agent_trace"] + [trace],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline() -> any:
    graph = StateGraph(AgentState)

    graph.add_node("router",    router_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("reasoning", reasoning_agent)
    graph.add_node("risk",      risk_agent)
    graph.add_node("synthesis", synthesis_agent)

    graph.set_entry_point("router")
    graph.add_edge("router",    "retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "risk")
    graph.add_edge("risk",      "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


# ── Run pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(query: str) -> AgentState:
    pipeline = build_pipeline()
    initial_state: AgentState = {
        "query":          query,
        "domain":         "",
        "retrieved_docs": [],
        "reasoning":      "",
        "risks":          [],
        "final_answer":   "",
        "agent_trace":    [],
        "current_agent":  "router",
        "error":          "",
    }
    result = pipeline.invoke(initial_state)
    return result


# ── CLI mode ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the key lessons on community ownership in rural road projects?"

    print(f"\n🔍 Query: {query}\n{'='*60}")
    result = run_pipeline(query)

    print(f"\n📋 AGENT TRACE:")
    for step in result["agent_trace"]:
        print(f"  [{step['agent']}] {step['duration_ms']}ms — {step['output_summary'][:80]}")

    print(f"\n⚠️  RISKS IDENTIFIED:")
    for r in result["risks"]:
        print(f"  [{r['severity']}] {r['risk']}")

    print(f"\n✅ FINAL ANSWER:\n{result['final_answer']}")
