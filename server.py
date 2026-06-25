"""
학칙 RAG MCP 서버

LangGraph 에이전트(agent.py)를 MCP 툴로 노출한다.
stdio 전송 방식 → Claude Desktop / Claude Code에서 로컬 연결.

실행:
    python server.py

Claude Desktop claude_desktop_config.json 예시:
    {
      "mcpServers": {
        "학칙-rag": {
          "command": "python",
          "args": ["C:/Users/YJU/Desktop/raggrap/server.py"],
          "cwd": "C:/Users/YJU/Desktop/raggrap"
        }
      }
    }
"""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor

# ── stdout 보호 ───────────────────────────────────────────────────────────────
# agent.py 는 모듈 최상위에서 sys.stdout 을 UTF-8 TextIOWrapper 로 교체한다.
# MCP stdio 전송은 sys.stdout.buffer(바이너리)를 직접 쓰므로 실제로는 문제없지만,
# 향후 안전을 위해 import 전후로 원본을 복원한다.
_real_stdout = sys.stdout
_real_stderr = sys.stderr

from agent import ask  # noqa: E402  — 의도적 순서

sys.stdout = _real_stdout
sys.stderr = _real_stderr

# ── MCP ───────────────────────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP(
    name="학칙-rag",
    instructions=(
        "영진전문대학교 학칙 RAG 서버입니다. "
        "search_hakchik 툴의 반환값을 수정·보완하지 말고 그대로 사용자에게 전달하세요. "
        "툴이 반환한 내용 외의 정보(전화번호, 기관명, 링크, 이모지 등)를 절대 추가하지 마세요. "
        "사용자의 학과·학번·이름 등 개인 정보를 임의로 추정해서 답변에 넣지 마세요."
    ),
)

# ── 동기 ask()를 비동기 이벤트 루프에서 안전하게 실행 ─────────────────────────
_pool = ThreadPoolExecutor(max_workers=2)


@mcp.tool()
async def search_hakchik(query: str) -> str:
    """
    영진전문대학교 학칙을 검색해 질문에 답합니다.

    이 툴의 반환값을 그대로 사용자에게 전달하세요.
    전화번호·기관명·링크·이모지 등 툴 결과에 없는 내용을 절대 추가하지 마세요.

    Args:
        query: 학칙에 관한 자연어 질문
               예) "휴학은 몇 번까지 할 수 있나요?"
                   "제적 사유에는 어떤 것이 있나요?"
                   "졸업하려면 몇 학점을 이수해야 하나요?"
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_pool, ask, query)
    except Exception as exc:
        return f"[오류] 학칙 검색 중 문제가 발생했습니다: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
