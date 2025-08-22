# make_linkedin_posts.py
# ---------------------------------------------------------------------------
# Generate multi-line LinkedIn post variations per video row in corpus_keywords.csv
#
# USAGE (PowerShell / Bash):
#   # Preferred: keep your key in .env (OPENAI_API_KEY=sk-xxx)
#   python make_linkedin_posts.py -c corpus_keywords.csv -o posts -n 5 -m gpt-5-nano `
#     -b BlockchainAdvisorsLtd --brand-url https://linkedin.com/company/blockchainadvisorsltd
#
#   # Or pass explicitly (overrides .env / env):
#   python make_linkedin_posts.py -c corpus_keywords.csv -o posts -n 5 -m gpt-5-nano `
#     -b BlockchainAdvisorsLtd --brand-url https://linkedin.com/company/blockchainadvisorsltd `
#     --api-key "sk-..."
#
#   # Debug: print masked key and its source (cli/env/dotenv)
#   python make_linkedin_posts.py -c corpus_keywords.csv --print-key
#
# ARGUMENTS / PARAMETERS
#   -c, --csv PATH
#       Path to input CSV. Must contain columns: file, text, keywords.
#       Typical value: corpus_keywords.csv (output from extract_keywords_srt.py)
#
#   -o, --outdir DIR
#       Output directory where per-item CSVs are written.
#       Each input row generates: posts/<basename>_posts.csv
#       Default: posts
#
#   -n, --num INT
#       Number of LinkedIn post variants to generate for each input row.
#       Recommended 4–5. Default: 5
#
#   -m, --model NAME
#       OpenAI chat model to use (cheap/fast models recommended).
#       Examples: gpt-5-nano, gpt-4o-mini
#       Default: gpt-5-nano
#
#   -b, --brand NAME
#       Brand handle (without @). Used in the dedicated follow line and hashtags.
#       Example: BlockchainAdvisorsLtd
#       Default: BlockchainAdvisorsLtd
#
#   --brand-url URL
#       Public LinkedIn Company Page URL for the brand.
#       Default: https://linkedin.com/company/blockchainadvisorsltd
#
#   --api-key KEY
#       OpenAI API key (highest precedence). If omitted, falls back to the
#       OPENAI_API_KEY environment variable, then to .env (repo root).
#
#   --print-key
#       Print the API key source and a masked key (first 4 + last 4 chars).
#
# BEHAVIOUR / OUTPUT
#   • Reads each row from the CSV and asks the model for N post variants.
#   • Writes one CSV per input row: <stem>_posts.csv with columns: variant, post.
#   • CSV is UTF-8 with BOM (Excel-friendly) and preserves hard line breaks.
#   • Each post is MULTI-LINE (one sentence/phrase per line), includes a dedicated follow line:
#       Follow @<Brand> — <BrandURL>
#     then a blank line, then all hashtags on a single final line (includes #<Brand>).
#   • Unicode punctuation is normalised to avoid mojibake (e.g., curly quotes/dashes).
# ---------------------------------------------------------------------------

import os, csv, json, argparse, pathlib, unicodedata, re
from openai import OpenAI

# ---- .env support (tries python-dotenv; falls back to a tiny parser) ----
def _load_env_from_dotenv(path: str = ".env") -> None:
    """
    Load environment variables from a .env file if present.
    Precedence: do NOT override variables already set in os.environ.
    """
    try:
        from dotenv import load_dotenv, find_dotenv  # type: ignore
        env_path = find_dotenv(filename=path, usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
            return
    except Exception:
        pass

    # Minimal fallback parser (KEY=VALUE, ignores comments/blank lines)
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', s)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1]
        if k not in os.environ:
            os.environ[k] = v

SYS_TMPL = """You are a senior B2B social media copywriter for LinkedIn.
Do NOT copy the transcript; synthesise and add value.

FORMAT RULES:
- Write MULTI-LINE copy: one sentence or short phrase per line (6–10 lines total).
- Include ONE dedicated line before hashtags:
  Follow @{brand} — {brand_url}
- Then ONE blank line, then all hashtags on ONE final line (6–9 total; include #{brand}).
- No emojis. No generic hype. 70–180 words. Each variant distinct."""

USR_TMPL = """Context:
- Keywords: {keywords}
- Transcript excerpt (reference only; do NOT copy): {excerpt}

Write EXACTLY {n} distinct LinkedIn posts following the FORMAT RULES.
Return strict JSON:
{{
  "posts": [
    {{"variant": 1, "text": "<post #1 with \\n line breaks>"}}{comma_tail}
  ]
}}"""

PUNCT_MAP = {
    ord('\u2010'): '-', ord('\u2011'): '-', ord('\u2012'): '-', ord('\u2013'): '-', ord('\u2014'): '-',
    ord('\u2212'): '-', ord('\u00A0'): ' ', ord('\u2009'): ' ', ord('\u202F'): ' ',
    ord('\u2018'): "'", ord('\u2019'): "'", ord('\u201C'): '"', ord('\u201D'): '"', ord('\u2026'): '...',
}

def sanitize_text(t: str) -> str:
    t = t.translate(PUNCT_MAP)
    t = unicodedata.normalize('NFKC', t)
    t = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', t)
    return t

def excerpt(text: str, limit=1200) -> str:
    return " ".join((text or "").split())[:limit]

def call_chat_json(client, model, system, user):
    kwargs = dict(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}])
    try:
        resp = client.chat.completions.create(response_format={"type":"json_object"}, **kwargs)
    except TypeError:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        if "response_format" in str(e).lower():
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    content = resp.choices[0].message.content if resp.choices else ""
    try:
        return json.loads(content)
    except Exception:
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(content[s:e+1])
        raise ValueError("Model returned non-JSON output.")

def ensure_format(t: str, brand: str, brand_url: str) -> str:
    t = t.replace("\r\n","\n").replace("\r","\n").strip()
    t = sanitize_text(t)

    if "\n#" in t:
        body, tags = t.split("\n#", 1)
        tags = "#" + tags.replace("\n", " ").strip()
    else:
        body, tags = t, ""

    follow_line = f"Follow @{brand} — {brand_url}"
    if f"@{brand.lower()}" not in body.lower():
        body = body.rstrip() + f"\n\n{follow_line}"
    elif follow_line not in body:
        if re.search(rf"(?mi)@{re.escape(brand)}", body):
            body = re.sub(rf"(?mi)^.*@{re.escape(brand)}.*$", follow_line, body, count=1)
        else:
            body = body.rstrip() + f"\n\n{follow_line}"

    body = body.rstrip()
    return body + ("\n\n" + tags if tags else "")

def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return (k[:2] + "..." + k[-2:]) if len(k) > 4 else "***"
    return k[:4] + "..." + k[-4:]

def main():
    # Capture whether OPENAI_API_KEY existed *before* loading .env
    pre_env_has_key = "OPENAI_API_KEY" in os.environ
    _load_env_from_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--csv", default="corpus_keywords.csv", help="Input CSV: file,text,keywords")
    ap.add_argument("-o","--outdir", default="posts", help="Output folder for per-item CSVs")
    ap.add_argument("-n","--num", type=int, default=5, help="Variants per item")
    ap.add_argument("-m","--model", default="gpt-5-nano", help="Model (e.g., gpt-5-nano, gpt-4o-mini)")
    ap.add_argument("-b","--brand", default="BlockchainAdvisorsLtd", help="Brand handle (no @)")
    ap.add_argument("--brand-url", default="https://linkedin.com/company/blockchainadvisorsltd", help="LinkedIn URL")
    ap.add_argument("--api-key", dest="api_key", default=None, help="OpenAI API key (overrides env/.env)")
    ap.add_argument("--print-key", action="store_true", help="Print masked API key and its source (cli/env/dotenv)")
    args = ap.parse_args()

    # Select key + infer source for debug
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if args.api_key:
        key_source = "cli"
    elif os.environ.get("OPENAI_API_KEY"):
        key_source = "env" if pre_env_has_key else "dotenv"
    else:
        key_source = "missing"

    if args.print_key:
        print(f"[make_linkedin_posts] API key source: {key_source}; key: {_mask_key(api_key or '')}")

    if not api_key:
        raise SystemExit(
            "OpenAI key missing. Provide --api-key, set OPENAI_API_KEY, or create .env with OPENAI_API_KEY=..."
        )

    client = OpenAI(api_key=api_key)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        fn   = (row.get("file") or "").strip()
        text = (row.get("text") or "").strip()
        kws  = (row.get("keywords") or "").strip()
        if not fn:
            continue

        sys_msg = SYS_TMPL.format(brand=args.brand, brand_url=args.brand_url)

        # JSON skeleton hint for n posts
        comma_tail = ""
        if args.num >= 2:
            placeholders = []
            for i in range(2, args.num + 1):
                placeholders.append('{{"variant": %d, "text": "<post #%d with \\\\n line breaks>"}}' % (i, i))
            comma_tail = ",\n    " + ",\n    ".join(placeholders)

        usr_msg = USR_TMPL.format(
            keywords=kws or "n/a",
            excerpt=excerpt(text),
            n=args.num,
            comma_tail=comma_tail
        )

        try:
            data = call_chat_json(client, args.model, sys_msg, usr_msg)
            posts = data.get("posts", [])
        except Exception as e:
            posts = [{"variant": 1, "text": f"[ERROR] {e}"}]

        out_path = pathlib.Path(outdir) / f"{pathlib.Path(fn).stem}_posts.csv"
        with open(out_path, "w", newline="", encoding="utf-8-sig") as wf:  # UTF-8 BOM for Excel
            w = csv.writer(wf, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["variant","post"])
            for p in posts:
                w.writerow([p.get("variant",""), ensure_format(p.get("text","") or "", args.brand, args.brand_url)])
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()