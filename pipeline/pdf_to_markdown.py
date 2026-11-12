"""
pdf_to_markdown.py — vision-LLM extraction of PDFs (slide decks with screenshots, scanned/garbled Thai PDFs)
into the same Markdown format as kb/*.md, ready for build_chunks.py.

Why: pdftotext/PyPDF only see the PDF text layer. In the MPVD manuals:
  * most content lives inside screenshots (switching rules, menus, steps) -> invisible to text extraction
  * the FC manual's text layer is corrupted (Thai fonts with WinAnsi encoding) -> pure garbage
  * Thai vowels/tone marks come out split ("เงือ ่ นไข") and "ำ" is lost ("ดาเนินการ")
A vision model reads the rendered page like a human does.

Usage:
  pip install openai anthropic pypdfium2 pillow
  export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY=...
  python pdf_to_markdown.py "MPVD/(HR_TH)Switching Manual Update.pdf" --doc-id mpvd_hr_switching \
      --title "คู่มือ Switching สำหรับ HR" --audience "HR / คณะกรรมการกองทุน" --out kb/hr_switching_manual.md
Options:
  --provider openai|anthropic   (default: openai if OPENAI_API_KEY is set)
  --model    e.g. gpt-4.1 / gpt-4o / claude-sonnet-4-5 (use a strong vision model; small ones misread Thai)
  --tiles    also send 2 zoomed halves of each page (recommended for slides with small screenshot text)
  --pages    e.g. 1-5,8

Always spot-check a few pages against the PDF afterwards (numbers, dates, account numbers).
"""
import argparse, base64, io, os, re, sys, time
from pathlib import Path

PROMPT = """You are converting page {page} of a Thai financial-services manual ("{title}") into Markdown for a RAG knowledge base.
The page image is attached{tiles_note}. The PDF's own text layer is given below ONLY as a spelling hint — it is often broken or incomplete; trust the image.

Rules:
1. Start with exactly one line: "## [p.{page}] <short descriptive heading in the page's language>". Use the slide/page title; if it is a continuation, say what step range it covers.
2. Transcribe ALL meaningful text verbatim, INCLUDING text inside screenshots, popups, tables, callout boxes and footnotes. Keep Thai spelling exact (ำ, tone marks). Keep numbers, dates, times, account numbers, URLs, phone numbers exactly.
3. For UI walkthroughs, write the numbered steps in order ("1. คลิกเมนู ..."), naming the exact button/menu labels in quotes, then list what the screen shows (fields, columns, options in dropdowns, warning texts).
4. Tables -> Markdown tables. Calendars/diagrams/flowcharts -> explain them in sentences that state the conclusion (e.g. "วันครบกำหนด = วันศุกร์ที่ 19 เพราะ ...").
5. Expand codes/abbreviations if the page defines them (e.g. "MRT-SF = นโยบายตราสารหนี้มั่นคง").
6. Skip decorative elements (logos, stock photos, page footers repeated on every page) — but keep a footer once if it contains contact info or a URL.
7. If the page is only a cover/divider, output the heading plus one line saying so.
8. Never invent content. If something is unreadable write [อ่านไม่ออก].
Output Markdown only, no preamble.

--- text layer hint ---
{hint}
"""


def render(pdf_path, page_no, scale=2.4):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    img = pdf[page_no - 1].render(scale=scale).to_pil().convert("RGB")
    hint = pdf[page_no - 1].get_textpage().get_text_range()
    return img, hint


def tiles(img):
    w, h = img.size
    if w >= h:   # landscape slide -> left/right halves with overlap
        return [img.crop((0, 0, int(w * .55), h)), img.crop((int(w * .45), 0, w, h))]
    return [img.crop((0, 0, w, int(h * .55))), img.crop((0, int(h * .45), w, h))]


def b64(img, max_side=2000):
    img = img.copy(); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_openai(model, prompt, images):
    from openai import OpenAI
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(i)}", "detail": "high"}} for i in images]
    r = OpenAI().chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content": content}])
    return r.choices[0].message.content


def call_anthropic(model, prompt, images):
    import anthropic
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(i, 1568)}} for i in images]
    content.append({"type": "text", "text": prompt})
    r = anthropic.Anthropic().messages.create(model=model, max_tokens=4096, temperature=0,
                                              messages=[{"role": "user", "content": content}])
    return r.content[0].text


def parse_pages(spec, n):
    if not spec:
        return list(range(1, n + 1))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--audience", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="openai" if os.getenv("OPENAI_API_KEY") else "anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--pages", default=None)
    a = ap.parse_args()
    model = a.model or ("gpt-4.1" if a.provider == "openai" else "claude-sonnet-4-5")
    call = call_openai if a.provider == "openai" else call_anthropic

    import pypdfium2 as pdfium
    n = len(pdfium.PdfDocument(a.pdf))
    sections = []
    for p in parse_pages(a.pages, n):
        img, hint = render(a.pdf, p)
        imgs = [img] + (tiles(img) if a.tiles else [])
        prompt = PROMPT.format(page=p, title=a.title, hint=(hint or "(empty)")[:4000],
                               tiles_note=" (first image = full page, next images = zoomed halves of the same page)" if a.tiles else "")
        for attempt in range(3):
            try:
                md = call(model, prompt, imgs).strip(); break
            except Exception as e:
                print(f"page {p} attempt {attempt+1} failed: {e}", file=sys.stderr); time.sleep(5 * (attempt + 1))
        else:
            md = f"## [p.{p}] (extraction failed)\n[อ่านไม่ออก]"
        md = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", md)
        if not md.startswith("## [p."):
            md = f"## [p.{p}] หน้า {p}\n{md}"
        sections.append(md)
        print(f"page {p}/{n} done ({len(md)} chars)")

    front = (f"---\ndoc_id: {a.doc_id}\ntitle: {a.title}\nsource_file: {Path(a.pdf).name}\n"
             f"audience: {a.audience}\nextracted_with: {a.provider}/{model}\n---\n\n# {a.title}\n\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(front + "\n\n".join(sections) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
