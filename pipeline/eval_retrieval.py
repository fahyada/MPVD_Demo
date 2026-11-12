"""
Compare retrieval quality: BASELINE (raw PDF text layer + fixed 1000/200 char splitter, i.e. the typical
"throw the PDF into a vector DB" setup) vs NEW (vision-transcribed Markdown + section chunks + context header).
Same embedding model for both (multilingual-e5-large), plus BM25 and hybrid (RRF) for the NEW index.
A hit = a retrieved chunk that comes from one of the gold (doc, page) pairs.
"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads/AI x CU/MPVD")
PDFS = {
    "mpvd_hr_switching": "(HR_TH)Switching Manual Update.pdf",
    "mpvd_member_switching": "(Member)Switching Manual - Update.pdf",
    "mpvd_line_connect": "LINE CONNECT Manual  V2025 update .pdf",
    "mpvd_fc_manual": "278_MRT_FC Manual_คู่มือคณะกรรมการกองทุน.pdf",
    "mpvd_epf_member_th": "คู่มือใช้บริการ_e-Provident Fund_Member_(TH)2026.pdf",
    "mpvd_epf_hr_manage_th": "TISCO_PVD_UserManual_ManageMembersContribution (7) (1).pdf",
    "mpvd_epf_hrfc_register_th": "TISCO_PVD_UserManual_HRFC_TH.pdf",
    "mpvd_myfunds_th": "คู่มือการใช้งาน MyFunds_PVD  (TH).pdf",
    "mpvd_myfunds_en": "คู่มือการใช้งาน MyFunds_PVD  (EN).pdf",
}
K = [1, 3, 5]

def baseline_chunks(size=1000, overlap=200):
    out = []
    for doc, fn in PDFS.items():
        n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(PDF_DIR / fn)], capture_output=True, text=True).stdout).group(1))
        # concatenate pages, remember char offsets -> page
        text, spans = "", []
        for p in range(1, n + 1):
            t = subprocess.run(["pdftotext", "-f", str(p), "-l", str(p), str(PDF_DIR / fn), "-"], capture_output=True, text=True).stdout
            spans.append((len(text), len(text) + len(t), p)); text += t
        i = 0
        while i < max(len(text), 1):
            seg = text[i:i + size]
            pages = {p for s, e, p in spans if s < i + size and e > i}
            if seg.strip():
                out.append({"id": f"{doc}_c{i}", "doc_id": doc, "pages": pages, "text": seg})
            i += size - overlap
    return out

def new_chunks():
    out = []
    for l in open(ROOT / "chunks.jsonl", encoding="utf-8"):
        c = json.loads(l)
        if c["type"] != "manual":
            continue  # evaluate PDF knowledge only (FAQ chunks would leak dataset questions)
        out.append({"id": c["id"], "doc_id": c["doc_id"], "pages": {c["page"]}, "text": c["text"]})
    return out

def hit(chunk, gold):
    return any(f"{chunk['doc_id']}:{p}" in gold for p in chunk["pages"])

def score(rankings, qs, chunks):
    res = {k: 0 for k in K}; mrr = 0
    for r, q in zip(rankings, qs):
        first = next((i for i, ci in enumerate(r[:10]) if hit(chunks[ci], set(q["gold"]))), None)
        for k in K:
            res[k] += first is not None and first < k
        mrr += 0 if first is None else 1 / (first + 1)
    n = len(qs)
    return {f"hit@{k}": round(res[k] / n, 3) for k in K} | {"MRR@10": round(mrr / n, 3)}

_cache = {}
def dense_rank(model, chunks, qs):
    key = (id(chunks), len(qs))
    if key in _cache:
        return _cache[key]
    D = np.array(list(model.embed(["passage: " + c["text"] for c in chunks], batch_size=4)))
    Q = np.array(list(model.embed(["query: " + q["q"] for q in qs], batch_size=8)))
    D /= np.linalg.norm(D, axis=1, keepdims=True); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    _cache[key] = [list(np.argsort(-s)) for s in Q @ D.T]
    return _cache[key]

def bm25_rank(chunks, qs):
    from pythainlp.tokenize import word_tokenize
    from rank_bm25 import BM25Okapi
    tok = lambda s: [w.lower() for w in word_tokenize(s, engine="newmm", keep_whitespace=False) if w.strip()]
    bm = BM25Okapi([tok(c["text"]) for c in chunks])
    return [list(np.argsort(-bm.get_scores(tok(q["q"])))) for q in qs]

def rrf(*rankings, k=60):
    out = []
    for rs in zip(*rankings):
        s = {}
        for r in rs:
            for i, ci in enumerate(r[:50]):
                s[ci] = s.get(ci, 0) + 1 / (k + i + 1)
        out.append(sorted(s, key=lambda c: -s[c]))
    return out

if __name__ == "__main__":
    qs = json.load(open(ROOT / "pipeline" / "eval_questions.json", encoding="utf-8"))
    model = TextEmbedding("intfloat/multilingual-e5-large", cache_dir=os.getenv("FASTEMBED_CACHE"), threads=os.cpu_count())
    base, new = baseline_chunks(), new_chunks()
    results = {}
    bd = dense_rank(model, base, qs); print("baseline embedded", flush=True)
    results["BASELINE raw text + fixed chunks (dense)"] = score(bd, qs, base)
    results["BASELINE (BM25)"] = score(bm25_rank(base, qs), qs, base)
    nd = dense_rank(model, new, qs); nb = bm25_rank(new, qs); print("new embedded", flush=True)
    results["NEW markdown + section chunks (dense)"] = score(nd, qs, new)
    results["NEW (BM25, Thai tokenized)"] = score(nb, qs, new)
    results["NEW hybrid dense+BM25 (RRF)"] = score(rrf(nd, nb), qs, new)
    # breakdown by where the answer lives
    for where in ["screenshot", "scanned"]:
        sub = [q for q in qs if q["where"].startswith(where)]
        idx = [i for i, q in enumerate(qs) if q["where"].startswith(where)]
        results[f"  baseline dense — answers in {where} ({len(sub)}q)"] = score([bd[i] for i in idx], sub, base)
        results[f"  NEW hybrid     — answers in {where} ({len(sub)}q)"] = score([rrf(nd, nb)[i] for i in idx], sub, new)
    print(f"{len(qs)} questions | baseline chunks={len(base)} | new chunks={len(new)}")
    for k, v in results.items():
        print(f"{k:55s} {v}")
    json.dump(results, open(ROOT / "eval_results.json", "w"), ensure_ascii=False, indent=1)
