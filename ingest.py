"""
학칙 PDF → pgvector 인덱싱 파이프라인 (Advanced RAG)

실행:
    python ingest.py
"""

import os
import sys
import io

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from scripts.chunk_preview import extract_lines, parse_bonchik, parse_buchik, extract_byeolpyo

load_dotenv()

PDF_PATH = "pdf/영진전문대학교 학칙.pdf"
COLLECTION_NAME = "hakchik_chunks"

CONNECTION_STRING = (
    f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)


def build_documents() -> list[Document]:
    """PDF를 파싱해 LangChain Document 리스트로 변환."""
    lines = extract_lines(PDF_PATH)
    chunks = parse_bonchik(lines) + parse_buchik(lines) + extract_byeolpyo(PDF_PATH)

    docs = []
    for c in chunks:
        docs.append(Document(
            page_content=c["text"],
            metadata={
                "chunk_id":   c["chunk_id"],
                "parent_id":  c.get("parent_id") or "",
                "chunk_type": c["chunk_type"],
                **c["metadata"],
            },
        ))
    return docs


def main():
    print("=== 학칙 인덱싱 시작 ===\n")

    # 1. 임베딩 모델
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 2. pgvector 벡터 스토어 (테이블 없으면 자동 생성)
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    # 3. 청크 생성
    print("PDF 파싱 중...")
    docs = build_documents()
    parent_cnt = sum(1 for d in docs if d.metadata["chunk_type"] == "parent")
    child_cnt  = sum(1 for d in docs if d.metadata["chunk_type"] == "child")
    print(f"  청크 생성 완료: 총 {len(docs)}개 (parent {parent_cnt} / child {child_cnt})\n")

    # 4. 기존 컬렉션 초기화 후 임베딩 저장
    print("기존 데이터 초기화...")
    vector_store.delete_collection()
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    print(f"임베딩 중... (OpenAI text-embedding-3-small, {len(docs)}개)")
    ids = vector_store.add_documents(docs)
    print(f"\n✓ 인덱싱 완료: {len(ids)}개 벡터 저장됨")
    print(f"  컬렉션: {COLLECTION_NAME}")
    print(f"  DB: {os.getenv('PG_DB')} @ {os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}")

    # 5. 간단한 검색 테스트
    print("\n=== 검색 테스트 ===")
    test_query = "휴학은 몇 번까지 할 수 있나요?"
    results = vector_store.similarity_search(test_query, k=3)
    print(f"질문: {test_query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.metadata.get('조_번호', r.metadata.get('별표_번호', ''))} "
              f"{r.metadata.get('조_제목', r.metadata.get('별표_제목', ''))}")
        print(f"    {r.page_content[:80]}...")
        print()


if __name__ == "__main__":
    main()
