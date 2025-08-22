#!/usr/bin/env python3
r"""
03_extract_keywords_srt.py — Build a single CSV of (file, text, keywords) from .srt files.

Examples
--------
# Current folder, non-recursive
python 03_extract_keywords_srt.py -i . -o corpus_keywords.csv -n 12

# Recurse subfolders under transcripts\
python 03_extract_keywords_srt.py -i .\transcripts -r -o corpus_keywords.csv -n 12

# Single file
python 03_extract_keywords_srt.py -i .\transcripts\10_audio.srt -o corpus_keywords.csv -n 12

Flags
-----
-i/--input   : .srt file or a directory
-o/--out     : Output CSV path (default: ./corpus_keywords.csv)
-r/--recursive : Recurse when input is a directory
-n/--keywords: Top keywords per file (default: 10)
"""

import argparse, csv, re, pathlib, numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}$")
DIGIT_LINE = re.compile(r"^\d+$")

BLOCKCHAIN_TERMS = [
    r"blockchain", r"web3", r"crypto", r"cryptocurrency", r"cryptocurrencies",
    r"defi", r"nft", r"nfts", r"token", r"tokens", r"ethereum", r"solana", r"bitcoin",
    r"btc", r"eth", r"erc20", r"erc\-20", r"erc721", r"erc\-721", r"smart contract", r"smart contracts",
    r"dapp", r"dapps", r"layer ?\d", r"gas fee(s)?"
]
BLOCKCHAIN_RE = re.compile(r"|".join([rf"\b{t}\b" for t in BLOCKCHAIN_TERMS]), re.IGNORECASE)

def collect_srts(inp: pathlib.Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() == ".srt":
            return inp.parent, [inp]
        raise SystemExit("Input file must be .srt")
    if inp.is_dir():
        pat = "**/*.srt" if recursive else "*.srt"
        return inp, sorted([p for p in inp.glob(pat) if p.is_file()])
    raise SystemExit("Input must be a .srt file or a directory containing .srt files.")

def srt_to_text(path: pathlib.Path) -> str:
    lines = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or DIGIT_LINE.match(line) or SRT_TIME.match(line):
                continue
            line = re.sub(r"<[^>]+>", " ", line)
            lines.append(line)
    text = " ".join(lines)
    text = BLOCKCHAIN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_vectorizer():
    extra = {"ve","ll","re","ah","yeah","okay","ok","mm","uh","um","like","got","get","go","going",
             "thing","things","really","actually","basically","kind","sort","use","used","using"}
    stops = sorted(set(ENGLISH_STOP_WORDS) | extra)
    def preproc(x: str) -> str:
        x = x.lower()
        x = BLOCKCHAIN_RE.sub(" ", x)
        return x
    return TfidfVectorizer(
        preprocessor=preproc, lowercase=True, stop_words=stops,
        ngram_range=(1,2), max_df=0.85, max_features=20000, norm="l2"
    )

def top_terms(row_vec, feature_names: np.ndarray, topn: int) -> List[str]:
    arr = row_vec.toarray().ravel()
    idx = np.argsort(arr)[::-1]
    out, seen = [], set()
    for i in idx:
        if arr[i] <= 0: break
        t = feature_names[i]
        if len(t) < 3: continue
        if BLOCKCHAIN_RE.search(t): continue
        if t in seen: continue
        seen.add(t); out.append(t)
        if len(out) >= topn: break
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract keywords from SRT into a single CSV.")
    ap.add_argument("-i","--input", required=True, help="Directory or single .srt file")
    ap.add_argument("-o","--out", default="corpus_keywords.csv", help="Output CSV path")
    ap.add_argument("-r","--recursive", action="store_true", help="Recurse when input is a directory")
    ap.add_argument("-n","--keywords", type=int, default=10, help="Keywords per file")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    root, files = collect_srts(in_path, args.recursive)
    if not files:
        print("No .srt files found."); return

    texts = []
    relnames = []
    for p in files:
        txt = srt_to_text(p) or "general topic"
        texts.append(txt)
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.name
        relnames.append(rel)

    vect = build_vectorizer()
    X = vect.fit_transform(texts)
    feats = np.array(vect.get_feature_names_out())

    rows = []
    for i, rel in enumerate(relnames):
        kws = top_terms(X[i], feats, args.keywords)
        rows.append({"file": rel, "text": texts[i], "keywords": ", ".join(kws)})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","text","keywords"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows).")

if __name__ == "__main__":
    main()
