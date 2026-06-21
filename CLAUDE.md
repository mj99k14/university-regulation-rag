# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어 설정

**모든 응답은 반드시 한국어로 작성한다.**

## 프로젝트 개요

학교 학칙 문서를 기반으로 한 **RAG(Retrieval-Augmented Generation) MCP 서버** 구현 프로젝트.

- 학칙 PDF/텍스트를 벡터 DB에 저장하고, MCP 툴로 노출하여 Claude가 학칙 내용을 검색·답변할 수 있게 한다.
- MCP 서버는 Claude Desktop 등에서 로컬로 연결하여 사용한다.

## 아키텍처

두 단계로 나뉜다: **인덱싱(Advanced RAG)** 과 **쿼리(Agentic RAG)**.

```
[인덱싱 파이프라인 - Advanced RAG]
학칙 문서 (PDF)
    ↓ 파싱 + 조(條) 단위 청킹
    ↓ 부모-자식 청크 계층화 (Parent-Child Chunking)
    ↓ 한국어 임베딩 모델
    ↓ 하이브리드 인덱스 (Dense + Sparse BM25)
pgvector (PostgreSQL)

[쿼리 파이프라인 - Agentic RAG]
사용자 질문
    ↓
LangGraph 에이전트 워크플로우
    ├─ [retrieve] pgvector 하이브리드 검색
    ├─ [grade]    검색 결과 관련성 평가 (LangChain)
    ├─ [rewrite]  질문 재작성 (검색 실패 시)
    └─ [generate] 최종 답변 생성
    ↓
MCP 서버 (Python, mcp 패키지, stdio 전송)
    ↓
Claude Desktop / Claude Code
```

### 레이어별 역할

| 레이어 | 기술 | 역할 |
|--------|------|------|
| 문서 파싱 | PyMuPDF / pdfplumber | 학칙 PDF → 텍스트 |
| 벡터 DB | PostgreSQL + pgvector | 임베딩 저장, 유사도 검색 |
| 인덱싱 | Advanced RAG | 계층적 청킹, 하이브리드 인덱스, 리랭킹 |
| 에이전트 | LangGraph + LangChain | 검색→평가→재작성→생성 루프 |
| MCP 서버 | `mcp` 패키지 | LangGraph 에이전트를 MCP 툴로 노출 |

## 빌드 / 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# PostgreSQL + pgvector 초기화 (최초 1회)
psql -U postgres -f sql/init_pgvector.sql

# 문서 인덱싱 (Advanced RAG 파이프라인)
python ingest.py --source docs/학칙.pdf

# MCP 서버 실행
python server.py
```

## 주요 설계 결정

- **청킹**: 조(條) 단위로 1차 분할 후, 조 전체를 부모 청크로 두고 항(項) 단위를 자식 청크로 저장 (Parent-Child).
- **하이브리드 검색**: pgvector의 벡터 검색(Dense)과 BM25(Sparse)를 앙상블하여 한국어 법령 특유의 정확한 용어 매칭을 보완.
- **에이전트 루프**: LangGraph로 retrieve → grade → (rewrite → retrieve)* → generate 상태 머신을 구성. 관련 문서가 없으면 질문을 재작성해 재검색한다.
- **임베딩 모델**: `jhgan/ko-sroberta-multitask` 또는 OpenAI `text-embedding-3-small` (한국어 성능 비교 후 결정).
- **MCP 전송**: 로컬 사용이므로 stdio 방식.

## 분기별 진행 기록

각 학기/분기가 끝날 때 `progress/` 디렉토리에 `YYYY-Q{N}.md` 형식으로 기록을 추가한다.

```
progress/
  2026-Q2.md   ← 현재 분기
  2026-Q3.md   ← 다음 분기 ...
```
