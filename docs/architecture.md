# 학칙 RAG 시스템 아키텍처

## 1. 전체 흐름

```
[PDF 문서]
    │
    ▼
extract_lines()          ← pdfplumber로 줄 추출 (표 페이지는 char 레벨 처리)
    │
    ├─ parse_bonchik()   ← 본칙 조문 파싱 (제1조 ~ 끝)
    ├─ parse_buchik()    ← 부칙 파싱 (개정일별)
    └─ extract_byeolpyo() ← 별표1, 별표2 파싱
    │
    ▼
Parent-Child 청크 생성
    │
    ▼
OpenAI text-embedding-3-small 임베딩
    │
    ▼
pgvector (PostgreSQL) 에 벡터 저장
    │
━━━━━━━━━━━━━━━━━ 인덱싱 끝 / 쿼리 시작 ━━━━━━━━━━━━━━━━━
    │
사용자 질문 (MCP 또는 직접 호출)
    │
    ▼
LangGraph 에이전트 (retrieve → grade → rewrite → generate)
    │
    ▼
최종 답변 → MCP 툴 반환 → Claude Desktop 출력
```

---

## 2. 청킹 방식

### 2-1. 계층 구조 (Parent-Child)

```
Parent (조 전체)
  └── Child (항 단위 ① ② ③)
        └── (호 1. 2. 3. 는 항 텍스트 안에 포함)

항이 없는 경우 (제21조 제적 등):
Parent (조 전체)
  └── Child (호 단위 1. 2. 3.)
```

### 2-2. 본칙 (parse_bonchik)

| 감지 패턴 | 처리 |
|---|---|
| `^부\s*칙` | 부칙 시작 → 본칙 파싱 종료 |
| `^■ \[별표 N\]` | 별표 시작 → 파싱 종료 |
| `^제N장 제목` | 장 번호/제목 갱신 (청크 생성 안 함) |
| `^제N조(제목) ...` | 새 조문 시작 → 이전 조 flush 후 누적 시작 |
| `^①②③...` | 항 시작 → hang_list에 추가 |

- 개정 태그 `<개정 날짜>`, `<신설 날짜>` 는 날짜 추출 후 텍스트에서 제거
- 각주 `[전문개정 ...]`, `[조신설 ...]` 등 제거
- 삭제 조문 (`삭제` 포함) 제외

**chunk_id 형식:**
- Parent: `제19조`, `제19조의2`
- Child: `제19조-항1`, `제19조-항2`
- 호 Child: `제21조-호1`, `제21조-호2`

### 2-3. 부칙 (parse_buchik)

- 개정일마다 여러 개의 부칙 섹션 존재
- `[<〈](\d{4}\.\d+\.\d+\.?)[>〉]` 패턴으로 날짜 추출 (ASCII `<>` + 유니코드 `〈〉` 모두 지원)
- 날짜 추출 실패 시 `부칙1`, `부칙2` 형식으로 폴백
- 청크 텍스트 앞에 `[부칙 2018.3.9]` 접두사 삽입 → 날짜로 벡터 검색 가능

**chunk_id 형식:**
- Parent: `부칙-2018.3.9-제1조`
- Child: `부칙-2018.3.9-제1조-항1`

### 2-4. 별표 (extract_byeolpyo)

| 별표 | 청킹 전략 |
|---|---|
| 별표1 (학위 종별) | Parent = 전체 텍스트, Child = `학위종 — 학과명` 행 |
| 별표2 (입학정원) | Parent = 전체 텍스트만 (컬럼 파싱 불안정으로 LLM에 위임) |

### 2-5. 표(Table) 추출

표가 있는 페이지는 `char` 레벨로 처리:
1. `pdfplumber.find_tables()` 로 표 영역 bbox 감지
2. 표 안 글자는 char 수집에서 제외 (깨진 텍스트 방지)
3. 표 밖 글자는 y좌표 ±3pt 기준으로 줄 묶음
4. `_table_to_rows()` 로 표 행 → 자연어 변환 (`헤더1 값1 / 헤더2 값2` 형식)
5. 병합 셀(None/빈값) → 바로 위 행 값 상속
6. y좌표 기준으로 텍스트줄 + 표행 통합 정렬

---

## 3. 검색 방식 (retriever.py)

### 3-1. Dense 검색 — 코사인 유사도

- **모델**: OpenAI `text-embedding-3-small`
- **DB**: PostgreSQL + pgvector
- **유사도 메트릭**: **코사인 유사도 (Cosine Similarity)**
  - pgvector 기본값: `vector <=>` 연산자 (코사인 거리)
  - LangChain `PGVector` 의 `as_retriever()` 가 내부적으로 사용
- **두 갈래로 운영**:

| 리트리버 | 필터 | 가중치 | 목적 |
|---|---|---|---|
| dense_child | `chunk_type: child` | 0.4 | 항 단위 정밀 매칭 |
| dense_parent | `chunk_type: parent` | 0.3 | 호(1.2.3.)만 있는 조항 보완 (예: 제21조 제적) |

### 3-2. Sparse 검색 — BM25

- **알고리즘**: BM25 (Best Match 25, TF-IDF 계열)
- **구현**: `langchain_community.retrievers.BM25Retriever`
- **가중치**: 0.3
- **목적**: 한국어 법령 특유의 정확한 용어 매칭 (예: "수업연한", "이수구분")
- 모든 청크(parent + child + 부칙 + 별표)를 대상으로 in-memory 인덱스 생성

### 3-3. 앙상블

```
EnsembleRetriever(
    retrievers = [dense_child, dense_parent, bm25],
    weights    = [0.4,         0.3,          0.3],
)
k = 8  (각 리트리버가 8개 → 중복 제거 후 최종 반환)
```

---

## 4. LangGraph 분기 설계 (agent.py)

### 4-1. 상태 (AgentState)

```python
class AgentState(TypedDict):
    query:          str           # 원래 질문
    rewrite_query:  str           # 재작성된 질문 (재검색용)
    documents:      list[Document]  # 검색 결과
    generation:     str           # 최종 답변
    retry_count:    int           # 재검색 횟수
```

### 4-2. 그래프 구조

```
[START]
    │
    ▼
retrieve ──────────────────────────────────────────────┐
    │                                                   │ (rewrite 후 재검색)
    ▼                                                   │
grade ──→ route_after_grade ──→ (documents 있음) ──→ generate ──→ [END]
                    │
                    └──→ (documents 없음, retry < 2) ──→ rewrite
                    │
                    └──→ (retry >= 2) ──────────────→ generate ──→ [END]
```

### 4-3. 각 노드

| 노드 | 역할 | LLM |
|---|---|---|
| `retrieve` | pgvector + BM25 하이브리드 검색, k=8 | — |
| `grade` | 각 청크가 질문에 관련 있는지 yes/no 판단 | Haiku 4.5 (×8 병렬) |
| `rewrite` | 검색 실패 시 법령 용어로 질문 재작성 | Haiku 4.5 |
| `generate` | 관련 조항 기반 최종 답변 생성 | Sonnet 4.6 |

### 4-4. 분기 함수 (route_after_grade)

```python
def route_after_grade(state) -> str:
    if state["documents"]:          # 관련 문서 있음
        return "generate"
    if state["retry_count"] >= 2:   # 재시도 초과
        return "generate"           # 빈 context로 "규정 없음" 답변
    return "rewrite"                # 질문 재작성 후 재검색
```

MAX_RETRY = 2 → 최대 3회 검색 (초기 1회 + 재검색 2회)

### 4-5. grade 병렬 처리

8개 청크를 순차적으로 평가하면 느리므로 `ThreadPoolExecutor`로 동시 실행:

```python
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(_grade_one, doc): doc for doc in state["documents"]}
    relevant = [
        doc for f in as_completed(futures)
        for doc, ok in [f.result()] if ok
    ]
```

효과: 8회 순차 Haiku 호출 → 1회 병렬 실행으로 약 7~8배 속도 향상

---

## 5. 유사도 메트릭 요약

| 방식 | 메트릭 | 특징 |
|---|---|---|
| Dense (pgvector) | **코사인 유사도** | 의미 기반, 문맥 이해, 한국어 동의어 처리 |
| Sparse (BM25) | **BM25 점수** | 정확한 키워드 매칭, 법령 용어에 강함 |
| 앙상블 | 가중 평균 | Dense 70% + BM25 30% (child:parent = 4:3) |

> **코사인 유사도**: 두 벡터 간 각도의 코사인 값 (0~1). 1에 가까울수록 유사.  
> **BM25**: 단어 빈도(TF) × 역문서 빈도(IDF)를 문서 길이로 정규화한 점수.
