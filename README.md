# Short video clips → Audio → Transcripts → Keywords → LinkedIn posts (Ubuntu 24.04)

End-to-end pipeline to:

1. extract audio tracks from `.mp4`,
2. transcribe/translate to `.srt/.vtt/.txt`,
3. mine keywords (TF-IDF),
4. generate multi-line **LinkedIn post variations** per item.

> Scripts are batch-friendly and avoid unnecessary re-encoding/decoding work.

---

## 0) What’s inside

* `01_extract_audio.py` — remuxes audio tracks out of MP4 **without re-encoding** (PyAV/FFmpeg bindings). Chooses `.m4a` for AAC/ALAC; otherwise `.mka`. Handles multiple audio streams, recursion, flat/mirrored folders.&#x20;
* `02_transcribe_whisper.py` — batch transcribe/translate with **openai-whisper**; decodes via **PyAV** (no `ffmpeg` subprocess), auto-selects **CUDA** if available, writes SRT/VTT/TXT.
* `03_extract_keywords_srt.py` — builds a single CSV `(file, text, keywords)` using **scikit-learn TF-IDF** over SRT text (timestamps/indices stripped), with domain-term filtering to keep keywords general.
* `04_make_linkedin_posts.py` — calls the OpenAI API to generate **N** distinct, **multi-line** LinkedIn posts per input row; enforces a dedicated **“Follow @Brand — URL”** line and a single final **hashtags** line. CSV outputs are Excel-friendly (UTF-8 BOM). API key via env or flag.
* **NEW**: `Makefile` — one-liners for install (CPU or CUDA), pipeline steps, GPU check, smoke test, clean/reset.
* **NEW**: `scripts/smoke_test.py` — validates FFmpeg/PyAV, Torch/CUDA, Whisper model load; optionally round-trips a sample media file.

---

## 1) Quickstart with **Make**

> Requires GNU Make (present by default on Ubuntu) and `sudo` for system deps.

```bash
# 1) System packages (Ubuntu 24.04)
make apt-deps

# 2) Create venv + install Python deps
#    Pick ONE based on your machine (CPU-only is universal):
make install-cpu
# or: make install-cu126
# or: make install-cu124
# or: make install-cu121

# 3) Environment checks
make gpu          # prints torch version, CUDA availability, device name
make smoke        # runs environment smoke test

# 4) End-to-end pipeline (override variables as needed)
make all IN=./videos MODEL=small NUM=5 \
  BRAND=BlockchainAdvisorsLtd \
  BRAND_URL=https://linkedin.com/company/blockchainadvisorsltd \
  OPENAI_KEY=sk-proj-....
```
or you can set OPENAI_API_KEY in the .env file and exclude it from the make all command above or include OPENAI_KEY in the above (notice the difference please!)

**Common Make variables** (override like `VAR=value`):
`IN` (input videos), `AUDIO_DIR`, `TRANS_DIR`, `CSV`, `POSTS_DIR`, `MODEL`, `FORMATS`, `TASK`, `DEVICE`, `NUM`, `BRAND`, `BRAND_URL`, `OPENAI_KEY`.

> Tip: for the posts step, set `OPENAI_API_KEY` in your shell (or pass `OPENAI_KEY=` to Make).

---

## 2) System prerequisites (Ubuntu 24.04)

Install build tools + FFmpeg dev headers (needed by **PyAV**):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg \
  build-essential pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswresample-dev libswscale-dev
```

> `02_transcribe_whisper.py` decodes via **PyAV** (not the `ffmpeg` CLI), but having `ffmpeg` on PATH is handy for diagnostics.&#x20;

---

## 3) Optional: NVIDIA GPU + CUDA + PyTorch

1. **Driver**:

```bash
nvidia-smi || echo "No NVIDIA driver yet"
# To install the recommended driver (then reboot):
# sudo ubuntu-drivers autoinstall && sudo reboot
```

2. **Choose the correct PyTorch wheel** (pick exactly one):

```bash
# CPU-only (portable)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. **Verify**:

```bash
python - << 'PY'
import torch
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

`02_transcribe_whisper.py` will default to `--device auto` and use CUDA+FP16 when available.&#x20;

---

## 4) Python environment

Create a per-project virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install packages:

```bash
# Core pipeline
pip install av openai-whisper numpy scikit-learn pandas openai

# Then install ONE torch wheel per Section 3 (CPU or CUDA)
```

---

## 5) Configuration (OpenAI API key)

For the posts generator:

```bash
export OPENAI_API_KEY="sk-..."   # preferred
# or pass --api-key "sk-..." to 04_make_linkedin_posts.py
```

The script normalises punctuation and line breaks; each variant includes a dedicated **“Follow @Brand — URL”** line and a single final **hashtags** line.&#x20;

---

## 6) Usage — step by step

### 6.1 Extract audio from `.mp4`

```bash
python 01_extract_audio.py -i ./videos -o ./audio -r
```

* Remux only (no re-encode); picks `.m4a` for AAC/ALAC or `.mka` otherwise.
* Multi-track files produce `*_audio-0.m4a`, `*_audio-1.m4a`, …
* `--flat` to avoid mirroring subfolders; `--force` to overwrite.

### 6.2 Transcribe / translate (Whisper)

```bash
python 02_transcribe_whisper.py -i ./audio -o ./transcripts -r \
  -m small -f srt,vtt,txt --task transcribe --device auto
```

* Models: `tiny|base|small|medium|large` (default `small`).
* Device: `auto|cpu|cuda` (default `auto` → CUDA if available).
* Writes any combination of SRT/VTT/TXT; decoding via **PyAV**.

### 6.3 Build corpus+keywords CSV

```bash
python 03_extract_keywords_srt.py -i ./transcripts -r \
  -o corpus_keywords.csv -n 12
```

* Strips indices/timestamps/markup, filters domain terms, TF-IDF (1–2-grams), outputs `file,text,keywords`.&#x20;

### 6.4 Generate LinkedIn post variations

```bash
export OPENAI_API_KEY="sk-..."
python 04_make_linkedin_posts.py -c corpus_keywords.csv -o posts -n 5 \
  -m gpt-5-nano -b BlockchainAdvisorsLtd \
  --brand-url https://linkedin.com/company/blockchainadvisorsltd
```

* One CSV per input row (`*_posts.csv`) with columns `variant,post`.
* Enforces **multi-line** copy, a dedicated **follow** line, then all hashtags on a single final line.&#x20;

---

## 7) End-to-end examples

### Folder of videos, recursive

```bash
# 1) Audio
python 01_extract_audio.py -i ./videos -o ./audio -r
# 2) Transcribe (auto GPU)
python 02_transcribe_whisper.py -i ./audio -o ./transcripts -r -m small -f srt,vtt,txt
# 3) Keywords
python 03_extract_keywords_srt.py -i ./transcripts -r -o corpus_keywords.csv -n 12
# 4) Posts
python 04_make_linkedin_posts.py -c corpus_keywords.csv -o posts -n 5 \
  -m gpt-5-nano -b BlockchainAdvisorsLtd --brand-url https://linkedin.com/company/blockchainadvisorsltd
```

### Single file, quick test

```bash
python 01_extract_audio.py -i ./videos/clip.mp4 -o ./audio
python 02_transcribe_whisper.py -i ./audio/clip_audio.m4a -o ./transcripts -m tiny
python 03_extract_keywords_srt.py -i ./transcripts/clip_audio.srt -o corpus_keywords.csv -n 10
python 04_make_linkedin_posts.py -c corpus_keywords.csv -o posts -n 4 --api-key "sk-..."
```

---

## 8) Performance tips

* Prefer smaller Whisper models (`small`/`base`) for speed; CUDA + FP16 (auto) accelerates significantly.&#x20;
* Keep inputs/outputs on SSD; avoid deep nesting unless needed.
* For multi-track `.mp4`, step 1 exports **all** audio streams, then step 2 transcribes the file(s) you care about.&#x20;

---

## 9) Troubleshooting

* **PyAV build/import** — ensure FFmpeg dev libs are installed; verify with:

  ```bash
  python -c "import av; import sys; print('PyAV OK')"
  ```
* **CUDA not used** — confirm `nvidia-smi` and that you installed a matching **cu121/cu124** torch wheel.
* **No audio** — confirm the file actually has an audio stream (`ffprobe -hide_banner file.mp4`) or run step 1 first.&#x20;
* **OpenAI auth** — set `OPENAI_API_KEY` or use `--api-key`; outbound TLS must be permitted.
* **Post format** — the generator enforces the follow line and consolidates hashtags onto a single final line.&#x20;

---

## 10) Make targets reference

* `apt-deps` — install Ubuntu packages (FFmpeg dev headers, etc.).
* `venv` — create `.venv` and upgrade pip/wheel/setuptools.
* `install-cpu` / `install-cu124` / `install-cu121` — install Python deps + chosen Torch wheel.
* `extract` / `transcribe` / `keywords` / `posts` — run each pipeline step.
* `all` — run the full pipeline.
* `gpu` — print Torch/CUDA info.
* `smoke` — run `scripts/smoke_test.py`.
* `clean` — remove outputs; `reset` — also remove the venv.

---

## 11) Repo layout

```
.
├─ 01_extract_audio.py
├─ 02_transcribe_whisper.py
├─ 03_extract_keywords_srt.py
├─ 04_make_linkedin_posts.py
├─ Makefile
└─ scripts/
   └─ smoke_test.py
```

---

## 12) Reproducibility

After a successful install (CPU or CUDA variant), lock versions:

```bash
pip freeze > requirements.txt
```

---

## 13) License & usage

* Ensure you own or are licensed to process the input media.
* API usage for post generation is billable (OpenAI). Keep `-n` modest and prefer lighter models.&#x20;

---

### Appendix: Quick CUDA wheel chooser

| Situation | Command                                                                |
| --------- | ---------------------------------------------------------------------- |
| CPU-only  | `pip install torch --index-url https://download.pytorch.org/whl/cpu`   |
| CUDA 12.6 | `pip install torch --index-url https://download.pytorch.org/whl/cu126` |
| CUDA 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

Verify:

```bash
python - << 'PY'
import torch; print(torch.__version__, torch.cuda.is_available())
PY
```