"""
학칙 PDF → Parent-Child 청크 JSON 미리보기 스크립트.

청킹 설계:
  - Parent : 조(條) 전체 텍스트
  - Child  : 항(①②③) 단위 텍스트  (호 1. 2. 3. 은 항 텍스트 안에 포함)
  - 별표   : 테이블 행 → 자연어 텍스트 청크

처리 규칙:
  - 부칙(附則) 섹션 제외
  - 삭제 조문(is_deleted=True) 제외
  - <개정 / <신설 태그 → 텍스트에서 제거, 최종 개정일은 메타데이터로 보존
  - [전문개정 / [조신설 / [제목개정 등 각주 → 제거
"""

import pdfplumber
import re
import json
import sys
import io

if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PDF_PATH = "pdf/영진전문대학교 학칙.pdf"
OUTPUT_PATH = "chunks_preview.json"

# ─── 정규식 ────────────────────────────────────────────────────────────────
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

HEADER_PAT  = re.compile(r"영진전문대학교 규정집 \[.*?\]")
DOC_HDR_PAT = re.compile(r"^(영진전문대학교 학칙|제정|개정|소관부서)")
JANG_PAT    = re.compile(r"^제(\d+)장\s+(.+)$")
JO_PAT      = re.compile(r"^(제(\d+)조(?:의(\d+))?)\((.+?)\)\s*(.*)")
HANG_PAT    = re.compile(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])(.*)")
BUCHIK_PAT  = re.compile(r"^부\s*칙")
BYEOLPYO_PAT = re.compile(r"^■ \[별표 (\d+)\]")

# 개정/신설 태그: <개정 날짜, 날짜, ...> 또는 <신설 날짜>
REVISION_TAG_PAT = re.compile(r"<(?:개정|신설|삭제)[^>]*>")
# 각주: [전문개정 ...] [조신설 ...] [제목개정 ...] [본조신설 ...] [조문이동 ...] 등
FOOTNOTE_PAT = re.compile(r"\[(?:전문개정|조신설|본조신설|제목개정|조문이동|항신설|항삭제|호신설|호삭제|신설|개정)[^\]]*\]")
# 날짜 추출: yyyy.mm.dd. 형식
DATE_PAT = re.compile(r"\d{4}\.\d{1,2}\.\d{1,2}\.")


def circled_to_num(c: str) -> int:
    return CIRCLED.index(c) + 1


def extract_latest_revision(text: str) -> str | None:
    """태그에서 날짜를 모두 추출해 가장 최신 날짜 반환."""
    tags = REVISION_TAG_PAT.findall(text)
    dates = []
    for tag in tags:
        dates.extend(DATE_PAT.findall(tag))
    if not dates:
        return None
    return max(dates)  # lexicographic max works for yyyy.mm.dd.


def clean_text(text: str) -> str:
    """개정 태그와 각주를 제거한 클린 텍스트 반환."""
    text = REVISION_TAG_PAT.sub("", text)
    text = FOOTNOTE_PAT.sub("", text)
    # 빈 줄 정리
    lines = [l.rstrip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ─── PDF 추출 ───────────────────────────────────────────────────────────────

def extract_lines(pdf_path: str):
    """줄 목록과 페이지 번호 반환. 반복 헤더 제거."""
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_i, page in enumerate(pdf.pages):
            raw = page.extract_text() or ""
            raw = HEADER_PAT.sub("", raw)
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    result.append((line, page_i + 1))
    return result


def extract_byeolpyo(pdf_path: str):
    """
    별표 섹션 Parent-Child 청킹.
      - 별표1 (학위 종별)  : parent = 별표1 전체, child = '종별 → 학과' 행
      - 별표2 (입학정원)   : parent = 별표2 전체, child = '계열/학과 수업연한 N년 입학정원 N명' 행
    raw 텍스트 기반 파싱 (테이블 추출 대신)
    """
    BYEOLPYO1_START = re.compile(r"별표 1\]")
    BYEOLPYO2_START = re.compile(r"별표 2\]")
    # 별표1: '학위종 계열/학과' 줄
    DEGREE_ROW = re.compile(r"^([\w가-힣]+전문학사|학사|공학기술교육인증.+)\s+(.+)$")
    # 별표2: '계열/학과명 숫자 숫자' 패턴 (수업연한·입학정원)
    DEPT_ROW = re.compile(r"^(.{4,30})\s+(\d)\s+(\d+)\s*$")

    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        byeolpyo_pages = {1: [], 2: []}  # 별표번호 → lines
        current_no = None

        for page_i in range(35, len(pdf.pages)):
            raw = HEADER_PAT.sub("", pdf.pages[page_i].extract_text() or "")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if BYEOLPYO1_START.search(line):
                    current_no = 1
                    continue
                if BYEOLPYO2_START.search(line):
                    current_no = 2
                    continue
                if current_no:
                    byeolpyo_pages[current_no].append(line)

    base_meta = {"문서명": "영진전문대학교 학칙", "section_type": "별표",
                 "is_deleted": False, "항_번호": None, "항_순서": None}

    # ── 별표1: 학위 종별 ──────────────────────────────────────────────────
    b1_lines = byeolpyo_pages[1]
    parent_text_1 = "\n".join(b1_lines)

    chunks.append({
        "chunk_id": "별표1",
        "parent_id": None,
        "chunk_type": "parent",
        "text": parent_text_1,
        "metadata": {**base_meta, "별표_번호": "별표1", "별표_제목": "학위의 종별"},
    })

    # child: '학위종별 — 학과명' 행 추출
    # 헤더/불필요 행 제외 패턴
    B1_SKIP = re.compile(
        r"^(전문학사\s*학위|학사학위|종\s*별|해당|비고|학점은행|전문기술석사\s*학위|"
        r"공학기술교육인증|학위과정|학위명|계열/학과|국문|영문|Mechanical|Electronic|"
        r"Electrical|Architectural|AS in|\<)"
    )
    # 학위 종별 감지: '공학전문학사', '간호학사', '전문기술석사' 등 (단독 행)
    DEGREE_PAT = re.compile(r"^([\w가-힣]+(전문학사|학사|석사))\s+(.*\S)")

    current_degree = None
    child_idx = 0
    for line in b1_lines:
        if not line or B1_SKIP.match(line):
            current_degree = None
            continue
        m = DEGREE_PAT.match(line)
        if m:
            current_degree = m.group(1)
            dept = m.group(3).strip()
            if dept:
                child_idx += 1
                chunks.append({
                    "chunk_id": f"별표1-{child_idx}",
                    "parent_id": "별표1",
                    "chunk_type": "child",
                    "text": f"{current_degree} — {dept}",
                    "metadata": {**base_meta, "별표_번호": "별표1",
                                 "별표_제목": "학위의 종별", "학위_종별": current_degree,
                                 "학과명": dept},
                })
        elif current_degree and line:
            child_idx += 1
            chunks.append({
                "chunk_id": f"별표1-{child_idx}",
                "parent_id": "별표1",
                "chunk_type": "child",
                "text": f"{current_degree} — {line}",
                "metadata": {**base_meta, "별표_번호": "별표1",
                             "별표_제목": "학위의 종별", "학위_종별": current_degree,
                             "학과명": line},
            })

    # ── 별표2: 입학정원 ──────────────────────────────────────────────────
    b2_lines = byeolpyo_pages[2]
    parent_text_2 = "\n".join(b2_lines)

    chunks.append({
        "chunk_id": "별표2",
        "parent_id": None,
        "chunk_type": "parent",
        "text": parent_text_2,
        "metadata": {**base_meta, "별표_번호": "별표2", "별표_제목": "모집단위별 입학정원 및 수업연한"},
    })

    # 별표2: 2025·2026 두 컬럼이 한 줄에 나란히 있어 행 단위 파싱이 불안정.
    # parent 청크만 두고, LLM이 전체 텍스트에서 직접 학과·정원 정보를 읽도록 함.

    return chunks


# ─── 본칙 파싱 ──────────────────────────────────────────────────────────────

def parse_bonchik(lines):
    doc_meta = {
        "문서명": "영진전문대학교 학칙",
        "문서_코드": "2-1-1",
        "소관부서": "교무처(교무팀)",
        "제정일": "1979-03-01",
        "최종개정일": "2026-02-10",
    }

    chunks = []

    jang_no, jang_title = None, None

    jo_id = None
    jo_no = None
    jo_ui = None
    jo_title = None
    jo_page = None
    jo_deleted = False
    jo_raw_lines = []   # 개정 태그 포함 원본 (날짜 추출용)

    hang_list = []       # [(hang_no, raw_text), ...]
    cur_hang_no = None
    cur_hang_raw = []

    in_buchik = False
    skip_doc_header = True

    def flush_hang():
        nonlocal cur_hang_no, cur_hang_raw
        if cur_hang_no is not None and cur_hang_raw:
            hang_list.append((cur_hang_no, "\n".join(cur_hang_raw)))
        cur_hang_no = None
        cur_hang_raw = []

    def flush_jo():
        nonlocal jo_id, jo_raw_lines, hang_list
        if jo_id is None:
            return
        flush_hang()

        raw_text = "\n".join(jo_raw_lines)
        if not raw_text.strip():
            return

        # 삭제 조문 제외
        if jo_deleted:
            jo_id = None
            jo_raw_lines.clear()
            hang_list.clear()
            return

        # 최신 개정일 추출 후 텍스트 정리
        latest_rev = extract_latest_revision(raw_text)
        parent_text = clean_text(raw_text)

        base_meta = {
            **doc_meta,
            "장_번호": f"제{jang_no}장" if jang_no else None,
            "장_제목": jang_title,
            "조_번호": jo_id,
            "조_번호_숫자": jo_no,
            "조_의": jo_ui,
            "조_제목": jo_title,
            "is_deleted": False,
            "section_type": "본문",
            "페이지": jo_page,
            "조항_최종개정일": latest_rev,
        }

        # Parent chunk
        chunks.append({
            "chunk_id": jo_id,
            "parent_id": None,
            "chunk_type": "parent",
            "text": parent_text,
            "metadata": {**base_meta, "항_번호": None, "항_순서": None},
        })

        # Child chunks
        for h_no, h_raw in hang_list:
            h_latest = extract_latest_revision(h_raw)
            h_text = clean_text(h_raw)
            chunks.append({
                "chunk_id": f"{jo_id}-항{h_no}",
                "parent_id": jo_id,
                "chunk_type": "child",
                "text": h_text,
                "metadata": {
                    **base_meta,
                    "항_번호": f"제{h_no}항",
                    "항_순서": h_no,
                    "조항_최종개정일": h_latest or latest_rev,
                },
            })

        jo_id = None
        jo_raw_lines.clear()
        hang_list.clear()

    for line, page_num in lines:
        # 부칙 시작 → 이후 전체 제외
        if BUCHIK_PAT.match(line):
            flush_jo()
            in_buchik = True

        if in_buchik:
            continue

        # 별표 섹션 → 별도 처리
        if BYEOLPYO_PAT.match(line):
            flush_jo()
            break

        # 문서 최상단 헤더 건너뜀
        if skip_doc_header:
            if DOC_HDR_PAT.match(line):
                continue
            skip_doc_header = False

        # 장 헤더
        m = JANG_PAT.match(line)
        if m:
            jang_no = int(m.group(1))
            # 장 제목에서도 개정 태그 제거
            jang_title = clean_text(m.group(2).strip())
            continue

        # 조 헤더
        m = JO_PAT.match(line)
        if m:
            flush_jo()

            jo_id    = m.group(1)
            jo_no    = int(m.group(2))
            jo_ui    = m.group(3)
            jo_title = m.group(4)
            jo_page  = page_num
            jo_deleted = "삭제" in (m.group(5) or "")

            rest = (m.group(5) or "").strip()
            jo_raw_lines = [f"{jo_id}({jo_title})", rest] if rest else [f"{jo_id}({jo_title})"]

            # 같은 줄에 항 마커가 있으면 항1 시작
            mh = HANG_PAT.match(rest)
            if mh:
                cur_hang_no = circled_to_num(mh.group(1))
                cur_hang_raw = [mh.group(2).strip()]
            continue

        if jo_id is None:
            continue

        # 항 시작
        m = HANG_PAT.match(line)
        if m:
            flush_hang()
            cur_hang_no = circled_to_num(m.group(1))
            cur_hang_raw = [m.group(2).strip()]
            jo_raw_lines.append(line)
            continue

        # 일반 줄 (조 본문 / 항 이어지는 줄)
        jo_raw_lines.append(line)
        if cur_hang_no is not None:
            cur_hang_raw.append(line)

    flush_jo()
    return chunks


# ─── 메인 ───────────────────────────────────────────────────────────────────

def main():
    lines = extract_lines(PDF_PATH)
    bonchik_chunks = parse_bonchik(lines)
    byeolpyo_chunks = extract_byeolpyo(PDF_PATH)

    chunks = bonchik_chunks + byeolpyo_chunks

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    parent_cnt = sum(1 for c in chunks if c["chunk_type"] == "parent")
    child_cnt  = sum(1 for c in chunks if c["chunk_type"] == "child")
    print(f"✓ 총 {len(chunks)}개 청크 생성")
    print(f"  - parent (조·별표 전체) : {parent_cnt}개")
    print(f"  - child  (항·별표 행)  : {child_cnt}개")
    print(f"✓ 저장 완료 → {OUTPUT_PATH}\n")

    # 샘플 출력
    samples = [
        next(c for c in chunks if c["chunk_id"] == "제3조"),
        next(c for c in chunks if c["chunk_id"] == "제3조-항1"),
        next(c for c in chunks if c["chunk_id"] == "별표1-1"),
        next(c for c in chunks if c["chunk_id"] == "별표2"),
    ]
    print("=== 샘플 청크 ===")
    for s in samples:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
