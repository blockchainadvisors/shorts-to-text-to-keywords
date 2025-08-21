# Short video clips → Audio → Transcripts → Keywords → Clusters (Windows)

End-to-end workflow to:

1. extract audio tracks from `1.mp4..58.mp4`,
2. transcribe to SRT/VTT/TXT,
3. build a CSV of corpus + keywords,
4. cluster by topics.

All commands are **PowerShell** on Windows.

---

## Prerequisites

* **PowerShell** (run as user).
* **FFmpeg** (either on `PATH` or copied next to your files).

  ```powershell
  winget install Gyan.FFmpeg  # or: choco install ffmpeg -y
  where ffmpeg; ffmpeg -version
  ```
* **Python 3.10+** and **pip**:

  ```powershell
  python --version
  ```
* Python packages:

  ```powershell
  pip install openai-whisper scikit-learn numpy
  ```

> If `whisper` can’t find FFmpeg, add its `bin` folder to `PATH` or run it from the same folder as `ffmpeg.exe`.

---

## Step 1 — Extract audio tracks from MP4

**Assumptions:** files are `1.mp4` … `58.mp4` in the current directory; FFmpeg/FFprobe are invoked as `.\ffmpeg` / `.\ffprobe`.

```powershell
1..58 | % {
  $num=$_; $f="$num.mp4";
  $cnt=(.\ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$f" | Measure-Object -Line).Lines
  if($cnt -gt 0){
    if($cnt -eq 1){
      .\ffmpeg -hide_banner -loglevel error -y -i "$f" -map 0:a:0 -c copy "${num}_audio.m4a"
    } else {
      0..($cnt-1) | % { $k=$_; .\ffmpeg -hide_banner -loglevel error -y -i "$f" -map 0:a:$k -c copy "${num}_audio-$k.m4a" }
    }
  }
}
```

**Output naming**

* Single track → `N_audio.m4a`
* Multiple tracks → `N_audio-0.m4a`, `N_audio-1.m4a`, …

---

## Step 2 — Transcribe audio (SRT/VTT/TXT)

```powershell
mkdir transcripts -Force
$m="small"     # tiny|base|small|medium|large
Get-ChildItem *.m4a | % {
  python -m whisper $_.FullName --model $m -f all --output_dir transcripts --verbose False
}
```

**Notes**

* Force language if needed: `--language en` (or `ro`, etc.).
* GPU (if CUDA): add `--device cuda`.
* Per-file example:

  ```powershell
  python -m whisper .\10_audio.m4a --model small -f all --output_dir transcripts --verbose False
  ```

---

## Step 3 — Extract keywords to a single CSV

Place the two scripts below in your working folder.

`extract_keywords_srt.py`

```python
# Usage: python extract_keywords_srt.py -i . -n 12
# Output: corpus_keywords.csv (columns: file,text,keywords)

import os, re, glob, csv, argparse
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}$")
DIGIT_LINE = re.compile(r"^\d+$")
BLOCKCHAIN_TERMS = [r"blockchain", r"web3", r"crypto", r"cryptocurrency", r"cryptocurrencies",
    r"defi", r"nft", r"nfts", r"token", r"tokens", r"ethereum", r"solana", r"bitcoin",
    r"btc", r"eth", r"erc20", r"erc\-20", r"erc721", r"erc\-721", r"smart contract", r"smart contracts",
    r"dapp", r"dapps", r"layer ?\d", r"gas fee(s)?"]
BLOCKCHAIN_RE = re.compile(r"|".join([rf"\b{t}\b" for t in BLOCKCHAIN_TERMS]), re.IGNORECASE)

def srt_to_text(path: str) -> str:
    lines=[]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line=raw.strip()
            if not line or DIGIT_LINE.match(line) or SRT_TIME.match(line): continue
            line=re.sub(r"<[^>]+>", " ", line)
            lines.append(line)
    text=" ".join(lines)
    text=BLOCKCHAIN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_vectorizer():
    extra={"ve","ll","re","ah","yeah","okay","ok","mm","uh","um","like","got","get","go","going",
           "thing","things","really","actually","basically","kind","sort","use","used","using"}
    stops=sorted(set(ENGLISH_STOP_WORDS)|extra)
    def preproc(x:str)->str:
        x=x.lower()
        x=BLOCKCHAIN_RE.sub(" ", x)
        return x
    return TfidfVectorizer(preprocessor=preproc, lowercase=True, stop_words=stops,
                           ngram_range=(1,2), max_df=0.85, max_features=20000, norm="l2")

def top_terms(row_vec, feature_names: np.ndarray, topn:int) -> List[str]:
    arr=row_vec.toarray().ravel()
    idx=np.argsort(arr)[::-1]
    out,seen=[],set()
    for i in idx:
        if arr[i] <= 0: break
        t=feature_names[i]
        if len(t)<3: continue
        if BLOCKCHAIN_RE.search(t): continue
        if t in seen: continue
        seen.add(t); out.append(t)
        if len(out)>=topn: break
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-i","--input", default=".", help="Folder with .srt files")
    ap.add_argument("-n","--keywords", type=int, default=10, help="Keywords per file")
    args=ap.parse_args()

    srts=sorted(glob.glob(os.path.join(args.input, "*.srt")))
    if not srts: print("No .srt files found."); return

    texts=[srt_to_text(p) or "general topic" for p in srts]
    vect=build_vectorizer(); X=vect.fit_transform(texts); feats=np.array(vect.get_feature_names_out())

    rows=[]
    for i,p in enumerate(srts):
        kws=top_terms(X[i], feats, args.keywords)
        rows.append({"file": os.path.basename(p), "text": texts[i], "keywords": ", ".join(kws)})

    with open("corpus_keywords.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["file","text","keywords"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote corpus_keywords.csv ({len(rows)} rows).")

if __name__=="__main__": main()
```

Run:

```powershell
python extract_keywords_srt.py -i . -n 12
```

---

## Step 4 — Cluster by keywords

`cluster_srt.py`

```python
# Usage: python cluster_srt.py -c corpus_keywords.csv -k auto -l 3
# Outputs: tags.csv, clusters.txt

import re, csv, argparse
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BLOCKCHAIN_TERMS=[r"blockchain",r"web3",r"crypto",r"cryptocurrency",r"cryptocurrencies",
    r"defi",r"nft",r"nfts",r"token",r"tokens",r"ethereum",r"solana",r"bitcoin",
    r"btc",r"eth",r"erc20",r"erc\-20",r"erc721",r"erc\-721",r"smart contract",r"smart contracts",
    r"dapp",r"dapps",r"layer ?\d",r"gas fee(s)?"]
BLOCKCHAIN_RE=re.compile(r"|".join([rf"\b{t}\b" for t in BLOCKCHAIN_TERMS]), re.IGNORECASE)

def build_text_vectorizer():
    extra={"ve","ll","re","ah","yeah","okay","ok","mm","uh","um","like","got","get","go","going",
           "thing","things","really","actually","basically","kind","sort","use","used","using"}
    stops=sorted(set(ENGLISH_STOP_WORDS)|extra)
    def preproc(x:str)->str:
        x=x.lower()
        x=BLOCKCHAIN_RE.sub(" ", x)
        return x
    return TfidfVectorizer(preprocessor=preproc, lowercase=True, stop_words=stops,
                           ngram_range=(1,2), max_df=0.85, max_features=20000, norm="l2")

def choose_k_auto(X,n):
    if n<=2: return n
    if n==3: return 2
    best_k,best=-1,-1
    for k in range(2, min(12,n-1)+1):
        km=KMeans(n_clusters=k, n_init=10, random_state=42)
        labels=km.fit_predict(X)
        if len(set(labels))<2: continue
        sc=silhouette_score(X, labels, metric="cosine")
        if sc>best: best, best_k = sc, k
    return best_k if best_k!=-1 else 1

def label_clusters(km, feats, top=3):
    out=[]
    for ci in range(km.n_clusters):
        center=km.cluster_centers_[ci]
        idx=np.argsort(center)[::-1]
        terms=[]
        for i in idx:
            t=feats[i]
            if len(t)<3 or BLOCKCHAIN_RE.search(t): continue
            if t not in terms: terms.append(t)
            if len(terms)>=top: break
        out.append(", ".join(terms) if terms else f"cluster-{ci}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-c","--csv", default="corpus_keywords.csv")
    ap.add_argument("-k","--clusters", default="auto")
    ap.add_argument("-l","--label_terms", type=int, default=3)
    ap.add_argument("--out_csv", default="tags.csv")
    ap.add_argument("--out_txt", default="clusters.txt")
    args=ap.parse_args()

    rows=[]; used_kw=False
    with open(args.csv,"r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            fn=(row.get("file") or "").strip()
            text=(row.get("text") or "").strip()
            kw=(row.get("keywords") or "").strip()
            doc=kw if kw else text
            doc=BLOCKCHAIN_RE.sub(" ", doc)
            if fn and doc: rows.append((fn, doc))
            if kw: used_kw=True
    if not rows: print("No rows found."); return

    names=[r[0] for r in rows]; docs=[r[1] if r[1].strip() else "general topic" for r in rows]
    vect=CountVectorizer(lowercase=True) if used_kw else build_text_vectorizer()
    X=vect.fit_transform(docs); feats=np.array(vect.get_feature_names_out()); n=len(names)

    k=choose_k_auto(X,n) if str(args.clusters).lower()=="auto" else max(1, min(int(args.clusters), n)); 
    if n==2 and k==1: k=2

    if k==1:
        labels=np.zeros(n, dtype=int); cluster_names=["general"]
    else:
        km=KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        labels=km.labels_; cluster_names=label_clusters(km, feats, args.label_terms)

    out=[["file","cluster_id","cluster_label"]]; groups=defaultdict(list)
    for i,fn in enumerate(names):
        cid=int(labels[i]); lab=cluster_names[cid] if cid<len(cluster_names) else f"cluster-{cid}"
        out.append([fn, cid, lab]); groups[lab].append(fn)

    with open(args.out_csv,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerows(out)
    with open(args.out_txt,"w",encoding="utf-8") as f:
        for lab,files in groups.items():
            f.write(f"[{lab}] ({len(files)} files)\n")
            for fn in files: f.write(f"  - {fn}\n"); f.write("\n")
    print(f"Done. {n} files, k={k}. Wrote {args.out_csv} and {args.out_txt}.")

if __name__=="__main__": main()
```

Run:

```powershell
python cluster_srt.py -c corpus_keywords.csv -k auto -l 3
```

**Outputs**

* `tags.csv` → `file, cluster_id, cluster_label`
* `clusters.txt` → readable groups with file lists

---

## Optional — Silence `joblib/loky` core warning

```powershell
$cores=(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum
$env:LOKY_MAX_CPU_COUNT = $cores
[Environment]::SetEnvironmentVariable("LOKY_MAX_CPU_COUNT","$cores","User")
```

---

## Troubleshooting

* **`FileNotFoundError` from whisper** → FFmpeg not on `PATH`. Add FFmpeg’s `bin` to `PATH` or run from that folder.
* **Execution policy blocks scripts** → run one-liners or:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
  ```
* **Small batches (1–3 files)** → `cluster_srt.py` auto-selects a valid `k` (no silhouette errors).
* **Non-English** → add `--language <code>` to whisper.
* **Performance** → try `--model medium`/`large` or `--device cuda` with a supported GPU.

---

## Quickstart (copy–paste)

```powershell
# 1) Extract audio
1..58 | % { $n=$_; $f="$n.mp4"; $c=(.\ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$f" | Measure-Object -Line).Lines; if($c -gt 0){ if($c -eq 1){ .\ffmpeg -hide_banner -loglevel error -y -i "$f" -map 0:a:0 -c copy "${n}_audio.m4a" } else { 0..($c-1) | % { $k=$_; .\ffmpeg -hide_banner -loglevel error -y -i "$f" -map 0:a:$k -c copy "${n}_audio-$k.m4a" } } } }

# 2) Transcribe (SRT/VTT/TXT)
mkdir transcripts -Force; $m="small"; Get-ChildItem *.m4a | % { python -m whisper $_.FullName --model $m -f all --output_dir transcripts --verbose False }

# 3) Keywords CSV
python extract_keywords_srt.py -i . -n 12

# 4) Clusters
python cluster_srt.py -c corpus_keywords.csv -k auto -l 3
```
