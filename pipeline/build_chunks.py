"""
build_chunks.py — turn the clean MPVD Markdown knowledge base into retrieval-ready chunks.

Input : kb/*.md  (one file per PDF, YAML front-matter + "## [p.N] heading" sections)
        optional: TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx  (FAQ / answer patterns)
Output: chunks.jsonl  — one JSON object per chunk:
        {id, type, doc_id, title, source_file, audience, page, heading,
         text   <- what you EMBED / index (has a context header prepended)
         content<- the raw section body (what you show the LLM / cite)
         image  <- relative path (under kb/) of the page picture, ONLY for pages where seeing it helps
                   (UI screenshots, calendars, diagrams); null otherwise}

Why this chunking:
  * One chunk = one page/section of the manual (each slide/page is one topic), so a step-by-step
    procedure never gets cut in half the way fixed 500-token windows cut it.
  * Every chunk carries a context header (document title + audience + section heading + page),
    so a chunk that only says "3. คลิก ยืนยัน" is still findable by "switching HR" queries and the
    LLM knows whether it is reading the HR or the Member procedure.
  * Long sections are split on paragraph/list boundaries (never mid-line) with the heading repeated.
  * Divider/cover pages with no real content are dropped.

Usage:
  python build_chunks.py --kb kb --out chunks.jsonl [--faq path/to/TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx]
"""
import argparse, json, re
from pathlib import Path

MAX_CHARS = 1400        # Thai chars per chunk body (≈ 500–700 tokens for most embedders)
MIN_BODY_CHARS = 60     # drop cover / divider pages shorter than this


def parse_front_matter(md: str):
    meta, body = {}, md
    m = re.match(r"^---\n(.*?)\n---\n", md, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = md[m.end():]
    return meta, body


def split_sections(body: str):
    """Yield (page:int|None, heading:str, text:str) for each '## [p.N] heading' block."""
    parts = re.split(r"^## ", body, flags=re.M)
    for part in parts[1:]:
        head, _, text = part.partition("\n")
        m = re.match(r"\[p\.(\d+)\]\s*(.*)", head.strip())
        page, heading = (int(m.group(1)), m.group(2)) if m else (None, head.strip())
        yield page, heading, text.strip()


def split_long(text: str, limit: int = MAX_CHARS):
    """Split on blank lines / list items, packing blocks up to `limit` chars."""
    if len(text) <= limit:
        return [text]
    blocks = re.split(r"\n(?=\n|\*\*|\d+[\.\)] |- |\| )", text)
    out, cur = [], ""
    for b in blocks:
        b = b.strip("\n")
        if not b:
            continue
        if cur and len(cur) + len(b) + 1 > limit:
            out.append(cur.strip())
            cur = b
        else:
            cur = f"{cur}\n{b}" if cur else b
    if cur.strip():
        out.append(cur.strip())
    return out


def context_header(meta, page, heading, part=None, n_parts=None):
    h = [f"เอกสาร: {meta.get('title', '')}"]
    if meta.get("audience"):
        h.append(f"สำหรับ: {meta['audience']}")
    sec = f"หัวข้อ: {heading}"
    if part is not None and n_parts and n_parts > 1:
        sec += f" (ส่วนที่ {part}/{n_parts})"
    h.append(sec)
    if page:
        h.append(f"หน้า: {page}")
    return "\n".join(h)


def manual_chunks(kb_dir: Path):
    img_dir = kb_dir / "images"   # kb/images/<doc_id>_p<page>.jpg exists only for pages where the picture helps
    for md_path in sorted(kb_dir.glob("*.md")):
        meta, body = parse_front_matter(md_path.read_text(encoding="utf-8"))
        doc_id = meta.get("doc_id", md_path.stem)
        for page, heading, text in split_sections(body):
            if len(re.sub(r"\(หน้าคั่นบท.*?\)|\s", "", text)) < MIN_BODY_CHARS:
                continue  # cover / divider page
            pieces = split_long(text)
            for i, piece in enumerate(pieces, 1):
                yield {
                    "id": f"{doc_id}_p{page}" + (f"_{i}" if len(pieces) > 1 else ""),
                    "type": "manual",
                    "doc_id": doc_id,
                    "title": meta.get("title"),
                    "source_file": meta.get("source_file"),
                    "audience": meta.get("audience"),
                    "page": page,
                    "heading": heading,
                    "image": (f"images/{doc_id}_p{page}.jpg" if (img_dir / f"{doc_id}_p{page}.jpg").exists() else None),
                    "text": context_header(meta, page, heading, i, len(pieces)) + "\n\n" + piece,
                    "content": piece,
                }


def faq_chunks(xlsx_path: Path):
    """One chunk per scenario: example questions + the most trustworthy answer.

    Answer priority: CS-corrected answer (col 9) > original email answer (col 8) > AI 'standard pattern' (col 10).
    The AI-generated standard answers contain some menu names / rules that do NOT match the manuals,
    so they are only used when nothing human-written exists, and are flagged.
    """
    import openpyxl
    ws = openpyxl.load_workbook(xlsx_path, data_only=True).active
    for row in ws.iter_rows(min_row=4, values_only=True):
        code, name = row[0], row[1]
        if not code or not name:
            continue
        questions = [str(q).strip() for q in row[2:7] if q and str(q).strip()]
        original, cs, std = (str(x).strip() if x else "" for x in row[7:10])
        cs_ok = cs and cs not in ("ไม่มีแก้ไข", "-")
        if cs_ok and original and len(cs) < 300:
            # CS column is only a short note/stub -> keep the original email answer too
            answer, src = f"{original}\n\n[หมายเหตุจากทีม CS] {cs}", "original_email+cs_note"
        elif cs_ok:
            answer, src = cs, "cs_corrected"
        elif original:
            answer, src = original, "original_email"
        elif std:
            answer, src = std, "ai_standard_unverified"
        else:
            continue
        q_block = "\n".join(f"- {q}" for q in questions)
        text = (f"คำถามที่พบบ่อย (FAQ) กองทุนสำรองเลี้ยงชีพ ทิสโก้ — หัวข้อ: {name}\n"
                + (f"ตัวอย่างคำถาม:\n{q_block}\n" if q_block else "")
                + f"\nแนวทางคำตอบ:\n{answer}")
        yield {
            "id": f"faq_{code}",
            "type": "faq",
            "doc_id": "mpvd_faq_dialogue_dataset",
            "title": "TISCO PVD AI Dialogue Dataset (FAQ / answer patterns)",
            "source_file": xlsx_path.name,
            "audience": "สมาชิกกองทุน",
            "page": None,
            "heading": f"{code} {name}",
            "answer_source": src,
            "text": text,
            "content": answer,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--out", default="chunks.jsonl")
    ap.add_argument("--faq", default=None)
    a = ap.parse_args()
    chunks = list(manual_chunks(Path(a.kb)))
    if a.faq:
        chunks += list(faq_chunks(Path(a.faq)))
    with open(a.out, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_type = {}
    for c in chunks:
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    lens = [len(c["content"]) for c in chunks]
    print(f"wrote {len(chunks)} chunks {by_type} -> {a.out}; body chars min/avg/max = "
          f"{min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")


if __name__ == "__main__":
    main()
