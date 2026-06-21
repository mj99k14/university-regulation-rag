"""
LangGraph 노드별 프롬프트 템플릿

- grade_prompt   : 검색 결과 관련성 평가 (yes/no)
- rewrite_prompt : 검색 실패 시 질문 재작성
- generate_prompt: 최종 답변 생성
"""

from langchain_core.prompts import ChatPromptTemplate

# ── 1. grade : 관련성 평가 ────────────────────────────────────────────────────
grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 학칙 검색 결과의 관련성을 평가하는 평가자입니다.\n"
     "사용자의 질문과 검색된 학칙 조항을 비교해 관련이 있으면 'yes', 없으면 'no'만 답하세요.\n"
     "다른 말은 하지 마세요."),
    ("human",
     "질문: {query}\n\n"
     "검색된 조항:\n{document}\n\n"
     "이 조항이 질문에 답하는 데 관련이 있습니까? (yes/no)"),
])

# ── 2. rewrite : 질문 재작성 ─────────────────────────────────────────────────
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 학칙 검색 전문가입니다.\n"
     "원래 질문의 의도를 유지하면서, 학칙에서 검색이 잘 되도록 질문을 재작성하세요.\n"
     "법령 용어(예: 수업연한, 이수구분, 제적, 휴학 등)를 활용하면 좋습니다.\n"
     "재작성된 질문만 출력하세요. 설명은 하지 마세요."),
    ("human", "원래 질문: {query}"),
])

# ── 3. generate : 최종 답변 생성 ─────────────────────────────────────────────
generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 영진전문대학교 학칙 전문 안내 도우미입니다.\n\n"
     "규칙:\n"
     "1. 반드시 아래 제공된 학칙 조항만을 근거로 답변하세요.\n"
     "2. 답변 시 관련 조항 번호(예: 제19조)를 반드시 명시하세요.\n"
     "3. 학칙에 명시되지 않은 내용은 '학칙에 규정되어 있지 않습니다'라고 답하세요.\n"
     "4. 친절하고 명확하게 한국어로 답변하세요.\n\n"
     "참고 학칙 조항:\n{context}"),
    ("human", "{query}"),
])
