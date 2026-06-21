# agent.py 설계 — Agentic RAG (LangGraph)

## 전체 흐름

```
사용자 질문
    │
    ▼
┌─────────────┐
│   retrieve  │  pgvector에서 유사 청크 검색 (하이브리드: Dense + BM25)
└──────┬──────┘
       │ 검색 결과
       ▼
┌─────────────┐
│    grade    │  검색 결과가 질문과 관련 있는지 LLM으로 평가
└──────┬──────┘
       │
   관련 있음? ──── NO ───▶ ┌─────────────┐
       │                   │   rewrite   │  질문을 다르게 재작성
       │                   └──────┬──────┘
       │                          │ (retrieve로 돌아가 재검색, 최대 2회)
       │
      YES
       │
       ▼
┌─────────────┐
│  generate   │  검색 결과 + 원본 질문으로 Claude가 최종 답변 생성
└──────┬──────┘
       │
       ▼
    최종 답변
```

---

## 노드별 상세 설명

### 1. `retrieve` — 하이브리드 검색

```python
# Dense 검색: pgvector 코사인 유사도
dense_results = vector_store.similarity_search(query, k=5)

# Sparse 검색: BM25 (정확한 법령 용어 매칭)
bm25_results = bm25_retriever.get_relevant_documents(query)

# 앙상블 (Dense 60% + BM25 40%)
results = ensemble_retriever.invoke(query)
```

**왜 하이브리드?**
학칙에는 "제적", "수업연한", "이수구분" 같은 정확한 법령 용어가 많음.
Dense만 쓰면 의미는 비슷하지만 다른 조항을 가져올 수 있음.
BM25가 정확한 키워드를 잡아주는 역할을 함.

---

### 2. `grade` — 관련성 평가

LLM에게 검색 결과가 질문과 관련 있는지 yes/no로 판단시킴.

```
프롬프트:
  질문: "{query}"
  검색된 문서: "{document}"
  
  이 문서가 질문에 답하는 데 관련이 있습니까? yes 또는 no로만 답하세요.
```

관련 없는 문서가 과반수면 → `rewrite`로 분기.

---

### 3. `rewrite` — 질문 재작성

원본 질문의 의도를 유지하면서 검색에 더 잘 걸리도록 표현을 바꿈.

```
예시:
  원본:  "학교 다니다 군대 가면 어떻게 해요?"
  재작성: "병역의무 이행 시 휴학 처리 방법 및 기간"
```

재작성 후 `retrieve`로 돌아가 재검색. 최대 2회 반복 후 강제 generate.

---

### 4. `generate` — 답변 생성

Claude(`claude-sonnet-4-6`)가 검색된 조항을 근거로 답변.

```
시스템 프롬프트:
  당신은 영진전문대학교 학칙 전문가입니다.
  반드시 아래 학칙 조항만을 근거로 답변하세요.
  조항 번호(예: 제19조)를 명시해 출처를 밝히세요.
  학칙에 없는 내용은 "학칙에 규정되어 있지 않습니다"라고 답하세요.
```

---

## LangGraph 상태(State) 정의

```python
class AgentState(TypedDict):
    query:        str           # 원본 질문
    rewrite_query: str          # 재작성된 질문 (없으면 query와 동일)
    documents:    list[Document] # 검색된 청크들
    generation:   str           # 최종 답변
    retry_count:  int           # 재검색 횟수 (최대 2)
```

---

## 파일 구조 (구현 예정)

```
agent.py          ← LangGraph 그래프 정의 + 노드 함수
retriever.py      ← 하이브리드 검색 (Dense + BM25) 초기화
prompts.py        ← grade / rewrite / generate 프롬프트 템플릿
```

---

## 구현 순서

1. `retriever.py` — pgvector + BM25 앙상블 리트리버
2. `prompts.py` — 3개 노드용 프롬프트
3. `agent.py` — LangGraph 그래프 조립
4. 단위 테스트: "휴학 몇 번?", "졸업 학점은?", "제적 사유는?" 등으로 검증
