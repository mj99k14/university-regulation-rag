"""
하이브리드 리트리버 (Dense + BM25 앙상블)

- Dense  : pgvector 코사인 유사도 (의미 기반)
- Sparse : BM25 (정확한 법령 키워드 매칭)
- 앙상블 : Dense 60% + BM25 40%

Parent-Child 전략:
  검색은 Child 청크(항 단위)로 수행 → 정밀도 향상
  결과에 Parent(조 전체) 추가 → 문맥 보완
"""

import os
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "pdf/영진전문대학교 학칙.pdf")
COLLECTION_NAME = "hakchik_chunks"
CONNECTION_STRING = (
    f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def _get_vector_store(embeddings: OpenAIEmbeddings | None = None) -> PGVector:
    return PGVector(
        embeddings=embeddings or _get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def _load_all_docs() -> tuple[list[Document], list[Document]]:
    """PDF 재파싱해서 (전체 문서, child만) 반환."""
    sys.path.insert(0, os.path.dirname(__file__))
    from scripts.chunk_preview import extract_lines, parse_bonchik, extract_byeolpyo

    lines = extract_lines(PDF_PATH)
    chunks = parse_bonchik(lines) + extract_byeolpyo(PDF_PATH)

    all_docs, child_docs = [], []
    for c in chunks:
        doc = Document(
            page_content=c["text"],
            metadata={
                "chunk_id":   c["chunk_id"],
                "parent_id":  c.get("parent_id") or "",
                "chunk_type": c["chunk_type"],
                **c["metadata"],
            },
        )
        all_docs.append(doc)
        if c["chunk_type"] == "child":
            child_docs.append(doc)

    return all_docs, child_docs


def get_hybrid_retriever(k: int = 5) -> EnsembleRetriever:
    """
    Dense + BM25 앙상블 리트리버 반환.

    k : 각 리트리버가 가져오는 청크 수 (앙상블 후 중복 제거됨)
    """
    embeddings = _get_embeddings()

    # ── Dense (pgvector) ─────────────────────────────────────────────────────
    # child 청크만 검색해 정밀도를 높임
    dense_retriever = _get_vector_store(embeddings).as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"chunk_type": "child"},
        }
    )

    # ── Sparse (BM25) ────────────────────────────────────────────────────────
    # BM25는 전체 청크 대상 (parent도 포함해 법령 용어 커버리지 확보)
    all_docs, _ = _load_all_docs()
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # ── 앙상블 ───────────────────────────────────────────────────────────────
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


def fetch_parent(child_doc: Document) -> Document | None:
    """
    Child 청크에서 Parent(조 전체) 청크를 DB에서 가져옴.
    grade 노드에서 문맥 보완이 필요할 때 사용.
    """
    parent_id = child_doc.metadata.get("parent_id")
    if not parent_id:
        return None

    store = _get_vector_store()
    results = store.similarity_search(
        "",  # 텍스트 검색 없이 메타데이터 필터만 사용
        k=1,
        filter={"chunk_id": parent_id},
    )
    return results[0] if results else None


# ── 단독 실행 시 검색 테스트 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    retriever = get_hybrid_retriever(k=4)

    queries = [
        "휴학은 몇 번까지 할 수 있나요?",
        "졸업하려면 몇 학점 이수해야 하나요?",
        "제적 사유가 뭔가요?",
    ]

    for q in queries:
        print(f"\n질문: {q}")
        docs = retriever.invoke(q)
        for i, d in enumerate(docs, 1):
            조 = d.metadata.get("조_번호", d.metadata.get("별표_번호", ""))
            제목 = d.metadata.get("조_제목", d.metadata.get("별표_제목", ""))
            print(f"  [{i}] {조} {제목} ({d.metadata.get('chunk_type')}) — {d.page_content[:60]}...")
