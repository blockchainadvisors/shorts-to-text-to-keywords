# ====== Config (override at CLI: make all IN=./videos MODEL=small NUM=5 BRAND=BlockchainAdvisorsLtd) ======
PYTHON      ?= python3
VENV_DIR    ?= .venv
PIP         := $(VENV_DIR)/bin/pip
PY          := $(VENV_DIR)/bin/python
ENV_FILE    ?= .env
# input folder with .mp4 files
IN          ?= ./videos
# output audio dir
AUDIO_DIR   ?= ./audio
# output transcripts dir
TRANS_DIR   ?= ./transcripts
# output corpus+keywords CSV
CSV         ?= corpus_keywords.csv
# output LinkedIn posts dir
POSTS_DIR   ?= ./posts

# whisper model: tiny|base|small|medium|large
MODEL       ?= small
# accepted formats: srt,vtt,txt
FORMATS     ?= srt
# or: translate
TASK        ?= transcribe
# auto|cpu|cuda
DEVICE      ?= auto
# variants per item for posts
NUM         ?= 5

BRAND       ?= BlockchainAdvisorsLtd
BRAND_URL   ?= https://linkedin.com/company/blockchainadvisorsltd
OPENAI_KEY  ?= $(OPENAI_API_KEY) # inherit from env by default

# ====== Virtualenv ======
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip wheel setuptools

# ====== Install (choose ONE of the install-* targets) ======
.PHONY: install-cpu
install-cpu: venv
	$(PIP) install av openai-whisper numpy scikit-learn pandas openai python-dotenv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu

.PHONY: install-cu126
install-cu126: venv
	$(PIP) install av openai-whisper numpy scikit-learn pandas openai python-dotenv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu126

.PHONY: install-cu124
install-cu124: venv
	$(PIP) install av openai-whisper numpy scikit-learn pandas openai python-dotenv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124

.PHONY: install-cu121
install-cu121: venv
	$(PIP) install av openai-whisper numpy scikit-learn pandas openai python-dotenv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu121

# ====== System deps (Ubuntu) ======
.PHONY: apt-deps
apt-deps:
	sudo apt update
	sudo apt install -y python3-venv python3-pip ffmpeg \
		build-essential pkg-config \
		libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
		libswresample-dev libswscale-dev

# ====== Pipeline steps ======
.PHONY: extract
extract:
	$(PY) 01_extract_audio.py -i "$(IN)" -o "$(AUDIO_DIR)" -r

.PHONY: transcribe
transcribe:
	$(PY) 02_transcribe_whisper.py \
		-i "$(AUDIO_DIR)" -o "$(TRANS_DIR)" -r \
		-m "$(MODEL)" -f "$(FORMATS)" --task "$(TASK)" --device "$(DEVICE)"

.PHONY: keywords
keywords:
	$(PY) 03_extract_keywords_srt.py -i "$(TRANS_DIR)" -r -o "$(CSV)" -n 12

.PHONY: posts
posts:
	$(PY) 04_make_linkedin_posts.py \
		--print-key \
		-c "$(CSV)" -o "$(POSTS_DIR)" -n "$(NUM)" -m gpt-5-nano \
		-b "$(BRAND)" --brand-url "$(BRAND_URL)"

# ====== One‑shot targets ======
.PHONY: all
all: extract transcribe keywords posts

.PHONY: gpu
gpu:
	@echo "Torch/CUDA check:"
	echo "import torch, sys\nprint('torch', torch.__version__)\nprint('cuda available:', torch.cuda.is_available())\nprint('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | $(PY)

.PHONY: smoke
smoke:
	$(PY) scripts/smoke_test.py

# ====== Housekeeping ======
.PHONY: clean
clean:
	rm -rf "$(AUDIO_DIR)" "$(TRANS_DIR)" "$(POSTS_DIR)" "$(CSV)"

.PHONY: reset
reset: clean
	rm -rf "$(VENV_DIR)"
