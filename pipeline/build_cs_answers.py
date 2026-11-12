"""
build_cs_answers.py — turn the "คำตอบแก้ไขจากทีม CS" column of TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx into an
exact-answer library (cs_answers.json). The bot returns these bodies VERBATIM (character for character) when a
question matches a scenario; the LLM only classifies, it never rewrites them.

Per scenario:
  questions : the 5 example questions (used for matching)
  mode      : "verbatim"  -> variants hold the exact CS text
              "original"  -> CS wrote "ไม่มีแก้ไข" (no change), so the original email answer is the approved text
              "escalate"  -> CS wrote a hand-off note (SC21): return it as is
              "no_text"   -> CS column only names cases without answer text (SC08): fall back to the manuals
  variants  : [{n, label (the case condition, NOT sent to the member), body (sent verbatim)}]
  cs_note   : instructions CS wrote for the AI that are not part of any answer (e.g. SC01 step 1)
"""
import json, re, sys
from pathlib import Path
import openpyxl

BODY_START = re.compile(r"^(เรียน|รียน|จากการตรวจสอบ)")
VARIANT = re.compile(r"^คำตอบที\s*่?\s*(\d+)\s*(.*)$")


def clean(s):
    # exact text, only trimmed at the ends; internal spacing/typos are kept on purpose
    return s.strip("\n").rstrip()


def parse_variants(cell):
    lines = cell.split("\n")
    idx = [i for i, l in enumerate(lines) if VARIANT.match(l.strip())]
    preamble = clean("\n".join(lines[: idx[0]])) if idx else ""
    out = []
    for k, i in enumerate(idx):
        block = lines[i: idx[k + 1] if k + 1 < len(idx) else len(lines)]
        m = VARIANT.match(block[0].strip())
        label_lines, j = [m.group(2).strip()], 1
        while j < len(block) and not BODY_START.match(block[j].strip()):
            if block[j].strip():
                label_lines.append(block[j].strip())
            j += 1
        out.append({"n": int(m.group(1)), "label": " ".join(x for x in label_lines if x),
                    "body": clean("\n".join(block[j:]))})
    return preamble, out


def main(xlsx, out_path):
    ws = openpyxl.load_workbook(xlsx, data_only=True).active
    lib = []
    for r in ws.iter_rows(min_row=4, values_only=True):
        code, name = r[0], r[1]
        if not code or not name:
            continue
        qs = [str(q).strip() for q in r[2:7] if q and str(q).strip()]
        original, cs = (r[7] or ""), (r[8] or "")
        entry = {"code": code, "name": name, "questions": qs, "cs_note": "", "variants": []}
        if cs.strip() == "ไม่มีแก้ไข":
            entry["mode"] = "original"
            entry["variants"] = [{"n": 1, "label": "", "body": clean(original)}]
        elif "ให้ AI ร่าง" in cs and not VARIANT.search(cs):
            entry["mode"] = "escalate"
            first, _, rest = cs.partition("\n")
            entry["cs_note"] = first.strip()
            entry["variants"] = [{"n": 1, "label": "", "body": clean(rest)}]
        else:
            pre, vs = parse_variants(cs)
            entry["cs_note"] = pre
            if vs and all(not v["body"] for v in vs):
                entry["mode"] = "no_text"
                entry["variants"] = vs
            elif vs:
                entry["mode"] = "verbatim"
                entry["variants"] = vs
            else:
                entry["mode"] = "verbatim"
                entry["variants"] = [{"n": 1, "label": "", "body": clean(cs)}]
        # every body must be an exact substring of the source cell
        src = original if entry["mode"] == "original" else cs
        for v in entry["variants"]:
            assert v["body"] in src, (code, v["n"])
        lib.append(entry)
    Path(out_path).write_text(json.dumps(lib, ensure_ascii=False, indent=1), encoding="utf-8")
    for e in lib:
        print(e["code"], e["mode"], [(v["n"], v["label"][:40], len(v["body"])) for v in e["variants"]],
              "| note:", e["cs_note"][:60].replace("\n", " "))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "TISCO_PVD_AI_Dialogue_Dataset_FINAL.xlsx",
         sys.argv[2] if len(sys.argv) > 2 else "cs_answers.json")
