"""에이전트 파이프라인 테스트 - 검색 품질 확인용"""
from dotenv import load_dotenv
load_dotenv()

from agent import graph

def test(query: str):
    print(f"\n{'='*60}")
    print(f"[질문] {query}")
    print("="*60)

    result = graph.invoke({
        "query": query,
        "rewrite_query": "",
        "documents": [],
        "generation": "",
        "retry_count": 0,
    })

    docs = result["documents"]
    print(f"\n[검색된 조항 수] {len(docs)}개")
    for i, d in enumerate(docs, 1):
        jo = d.metadata.get("조_번호", d.metadata.get("별표_번호", ""))
        title = d.metadata.get("조_제목", d.metadata.get("별표_제목", ""))
        print(f"  [{i}] {jo} {title}")

    retry = result.get("retry_count", 0)
    if retry > 0:
        print(f"\n[재작성된 질문] {result.get('rewrite_query', '')} (재시도 {retry}회)")

    print(f"\n[답변]\n{result['generation']}")


if __name__ == "__main__":
    test("휴학은 몇 번까지 할 수 있나요?")
    test("제적 사유가 뭔가요?")
    test("졸업하려면 몇 학점 이수해야 하나요?")
