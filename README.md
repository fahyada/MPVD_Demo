# MPVD knowledge base — better PDF ingestion for the TISCO PVD chatbot

## Why the bot couldn't answer
Checked all 4 MPVD PDFs:

| PDF | Pages | What the text layer gives you | Problem |
|---|---|---|---|
| (HR_TH) Switching Manual | 7 | ~1.9k chars, mostly callout labels ("1.คลิก", "2.เลือก") | Rules, dates, policy codes, dropdown options exist **only inside screenshots** |
| (Member) Switching Manual | 14 | ~2.5k chars | Same; switching conditions, option cards and statuses are images |
| LINE CONNECT Manual | 23 | ~6k chars | Registration fields, OTP rules and the Fund Fact Sheet timing are images |
| FC Manual (คู่มือคณะกรรมการกองทุน) | 21 | ~20k chars of **garbage** (`ก ก !" # $`) | Printed through Ghostscript with Thai fonts in WinAnsi encoding, so every Thai character is lost. The biggest, most rule-heavy doc was effectively empty in the vector DB |

| e-Provident Fund Member manual (TH) 2026 | 57 | ~8k chars, 23 pages near-empty | Registration, Reference ID, OTP, password-reset rules are all screenshots |
| ManageMembersContribution (HR/FC, TH p1–55 + EN p56–109) | 109 | ~12k chars | Member add/edit/resign/transfer flows are screenshots |
| UserManual_HRFC_TH | 9 | ~1.4k chars | Registration screens |
| MyFunds_PVD (TH / EN) | 9 / 7 | <1k chars | App screenshots |

On top of that, `pdftotext`/PyPDF splits Thai vowels and tone marks (`เงือ ่ นไข`) and drops `ำ` (`ดำเนินการ` → `ดาเนินการ`), which hurts both embeddings and keyword search.

## What's in this folder
```
kb/                         clean Markdown, one file per PDF (ManageMembers split TH/EN), "## [p.N] heading" per page
  hr_switching_manual.md        member_switching_manual.md     line_connect_manual.md
  fc_manual.md                  epf_member_manual_th.md        epf_hrfc_register_th.md
  epf_hr_manage_members_th.md   epf_hr_manage_members_en.md    myfunds_app_th.md   myfunds_app_en.md
  images/                     197 page pictures (only pages where seeing the screen helps)
chunks.jsonl                279 retrieval-ready chunks (269 manual + 10 FAQ), 226 carry an `image` → load into your vector DB
demo/                       the published demo page (mpvd_assistant.html + template.html)
pipeline/
  build_chunks.py           kb/*.md (+ dialogue xlsx) → chunks.jsonl
  pdf_to_markdown.py        vision-LLM extractor for NEW PDFs (OpenAI or Anthropic key)
  eval_questions.json       41 test questions whose answers live in the PDFs (with gold pages)
  eval_retrieval.py         baseline vs new retrieval benchmark
eval_results.json
```

## Benchmark (59 questions answered in the 9 PDFs, multilingual-e5-large, same model both sides)
| Setup | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| Old: raw PDF text + fixed 1000/200 chunks | 0.56 | 0.66 | 0.68 |
| New: clean Markdown + page/section chunks (dense) | 0.78 | 0.92 | 0.97 |
| New: hybrid dense + Thai BM25 (RRF) | **0.81** | **0.93** | **0.98** |
| Old, FC-manual questions only (17) | 0.29 | 0.35 | 0.35 |
| New hybrid, FC-manual questions only | 0.82 | 0.88 | **1.00** |
| Demo page's in-browser search (BM25 words + bigrams) | 0.73 | – | 0.98 |

The old numbers are generous: a "hit" only means the retrieved chunk came from the right page. For the FC manual those chunks are garbled characters, and for the slide decks they mostly hold callout labels, so even a "hit" usually gave the LLM nothing to answer from. Caveat: I wrote the test questions after reading the documents, so they may lean slightly toward the new Markdown's wording. Add your own real user questions to `eval_questions.json`.

## How to plug it into the existing bot
Replace the "load PDF → split → embed" step with:

```python
import json
chunks = [json.loads(l) for l in open("chunks.jsonl", encoding="utf-8")]
for c in chunks:
    vector_db.add(
        id=c["id"],
        text_to_embed=c["text"],          # has a context header: doc title, audience, section, page
        payload={                         # what you give the LLM + use for citations
            "content": c["content"], "title": c["title"], "page": c["page"],
            "doc_id": c["doc_id"], "type": c["type"], "source_file": c["source_file"],
        },
    )
```
Retrieval settings that worked best in the benchmark:
1. **Hybrid search** (recommended; it beat dense-only at every cutoff once all 9 PDFs were in): dense embeddings + BM25 over Thai-tokenized text (`pythainlp.word_tokenize(engine="newmm")`), merged with Reciprocal Rank Fusion. Thai questions often hinge on exact terms (MRT-SF, vesting, 11.00 น.) that BM25 catches and embeddings blur.
2. **A multilingual embedder** (e.g. `multilingual-e5-large`, `bge-m3`, or OpenAI `text-embedding-3-large`). For e5, prefix with `query:` / `passage:`.
3. **top-k = 5**. Chunks are page-sized, so 5 chunks ≈ 4k Thai characters.
4. In the answer prompt, show `title` + `page` with each chunk and ask the model to cite them. The HR and Member switching manuals overlap, so the title tells the model which audience a step is for.

## Notes on the FAQ chunks (TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx)
* One chunk per scenario (SC01–SC08, SC21, SC22): the 5 example questions plus the answer.
* Answer priority: **CS-corrected answer > original email answer > AI "standard pattern" answer.** The blue AI-generated "standard" answers contain claims that don't match the manuals. For example, SC08 gives the menu as 'บริการสมาชิก' → 'เปลี่ยนแปลงทางเลือกการลงทุน (Switching)', but the manuals say 'เปลี่ยนนโยบาย / ทางเลือกการลงทุน'; SC06 says money can be kept "คราวละ 1 ปี", which the FC manual doesn't state. Don't index that column as truth.
* Don't use the dataset's example questions to *evaluate* retrieval. They are inside the FAQ chunks, so the scores would be inflated. Use `pipeline/eval_questions.json` instead.

## Adding more PDFs (e.g. the MF fund docs later)
```bash
pip install openai pypdfium2 pillow   # or anthropic
export OPENAI_API_KEY=...
python pipeline/pdf_to_markdown.py "path/to/file.pdf" --doc-id my_doc --title "..." --audience "..." --tiles --out kb/my_doc.md
python pipeline/build_chunks.py --kb kb --out chunks.jsonl --faq "MPVD/TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx"
python pipeline/eval_retrieval.py "path/to/MPVD"     # needs: pip install fastembed pythainlp rank_bm25
```
Use a strong vision model (gpt-4.1 / gpt-4o / Claude Sonnet). Small models misread Thai. Spot-check numbers, dates and account numbers afterwards; that's how the kb/ files here were made.
