"""
Agentic RAG — LangGraph 상태 머신

흐름: retrieve → grade → (rewrite → retrieve)* → generate
최대 재검색 2회 후 강제 generate
"""

import os
import sys
import io
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from retriever import get_hybrid_retriever
from prompts import grade_prompt, rewrite_prompt, generate_prompt

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MAX_RETRY = 2

# ── 상태 정의 ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query:         str
    rewrite_query: str
    documents:     list[Document]
    generation:    str
    retry_count:   int


# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",   # 빠른 grade/rewrite용
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

llm_generate = ChatAnthropic(
    model="claude-sonnet-4-6",           # 최종 답변 품질용
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

retriever = get_hybrid_retriever(k=8)


# ── 노드 함수 ─────────────────────────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    """pgvector + BM25 하이브리드 검색."""
    q = state.get("rewrite_query") or state["query"]
    docs = retriever.invoke(q)
    return {"documents": docs}


def grade(state: AgentState) -> dict:
    """검색 결과 관련성 평가. 관련 문서가 없으면 retry_count 증가."""
    chain = grade_prompt | llm
    relevant = []
    for doc in state["documents"]:
        result = chain.invoke({
            "query": state["query"],
            "document": doc.page_content,
        })
        if result.content.strip().lower().startswith("yes"):
            relevant.append(doc)

    return {
        "documents": relevant,
        "retry_count": state.get("retry_count", 0) + (0 if relevant else 1),
    }


def rewrite(state: AgentState) -> dict:
    """관련 문서가 없을 때 질문을 재작성해 재검색 준비."""
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": state["query"]})
    return {"rewrite_query": result.content.strip()}


def generate(state: AgentState) -> dict:
    """검색된 조항을 근거로 Claude Sonnet이 최종 답변 생성."""
    context = "\n\n".join(
        f"[{d.metadata.get('조_번호', d.metadata.get('별표_번호', ''))} "
        f"{d.metadata.get('조_제목', d.metadata.get('별표_제목', ''))} "
        f"(p.{d.metadata.get('페이지', '?')})]\n{d.page_content}"
        for d in state["documents"]
    )
    if not context:
        context = "관련 학칙 조항을 찾지 못했습니다."

    chain = generate_prompt | llm_generate
    result = chain.invoke({"context": context, "query": state["query"]})
    return {"generation": result.content}


# ── 라우팅 ────────────────────────────────────────────────────────────────────

def route_after_grade(state: AgentState) -> str:
    """관련 문서가 있으면 generate, 없으면 rewrite(최대 MAX_RETRY회)."""
    if state["documents"]:
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRY:
        return "generate"   # 재시도 초과 → 그냥 답변
    return "rewrite"


# ── 그래프 조립 ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade",    grade)
    g.add_node("rewrite",  rewrite)
    g.add_node("generate", generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite":  "rewrite",
    })
    g.add_edge("rewrite",  "retrieve")
    g.add_edge("generate", END)

    return g.compile()


graph = build_graph()


def ask(query: str) -> str:
    """외부에서 호출하는 단일 인터페이스."""
    result = graph.invoke({
        "query":         query,
        "rewrite_query": "",
        "documents":     [],
        "generation":    "",
        "retry_count":   0,
    })
    return result["generation"]


# ── 단독 실행 시 대화형 테스트 ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 학칙 RAG Agent (종료: q) ===\n")
    while True:
        q = input("질문: ").strip()
        if q.lower() == "q":
            break
        if not q:
            continue
        print("\n답변 생성 중...\n")
        print(ask(q))
        print()
