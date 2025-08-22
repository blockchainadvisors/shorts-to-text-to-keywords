#!/usr/bin/env python3
"""
Smoke test for the pipeline environment.

Checks:
- FFmpeg present
- PyAV import & version
- Whisper import & tiny model load
- Torch import & CUDA visibility
- Optional: if you pass a media path, confirm script #1 and #2 run.

Usage:
  python scripts/smoke_test.py
  python scripts/smoke_test.py path/to/sample.mp4
"""
import os, shutil, subprocess, sys, pathlib, json, tempfile

def ok(msg):  print(f"[OK] {msg}")
def bad(msg): print(f"[FAIL] {msg}"); sys.exit(1)
def info(msg): print(f"[..] {msg}")

def which(bin_name):
    return shutil.which(bin_name) is not None

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    py = pathlib.Path(sys.executable)

    info("Checking FFmpeg...")
    if which("ffmpeg"):
        code, out = run(["ffmpeg", "-version"])
        if code == 0:
            ok("ffmpeg found")
        else:
            bad("ffmpeg present but not runnable")
    else:
        info("ffmpeg not found on PATH (Whisper via PyAV can still work)")

    info("Checking PyAV import...")
    try:
        import av
        ok(f"PyAV {av.__version__}")
    except Exception as e:
        bad(f"PyAV import failed: {e}")

    info("Checking Torch & CUDA...")
    try:
        import torch
        ok(f"torch {torch.__version__} (cuda={torch.cuda.is_available()})")
        if torch.cuda.is_available():
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        bad(f"Torch import failed: {e}")

    info("Checking Whisper import & tiny model...")
    try:
        import whisper
        model = whisper.load_model("tiny", device="cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")
        ok("whisper tiny loaded")
    except Exception as e:
        bad(f"Whisper load failed: {e}")

    # Optional sample processing if a path is given
    if len(sys.argv) > 1:
        sample = pathlib.Path(sys.argv[1]).resolve()
        if not sample.exists():
            bad(f"Sample not found: {sample}")
        info(f"Running step #1 extract on: {sample}")
        code, out = run([str(py), "01_extract_audio.py", "-i", str(sample), "-o", "audio_tmp"])
        print(out)
        if code != 0:
            bad("Step #1 failed")
        ok("Step #1 completed")

        # Find first produced audio file
        audio_dir = repo / "audio_tmp"
        audio_files = list(audio_dir.glob("**/*.*"))
        if not audio_files:
            info("No audio files produced (video may have no audio). Skipping step #2.")
            return 0
        audio_file = audio_files[0]
        info(f"Running step #2 transcribe on: {audio_file}")
        code, out = run([str(py), "02_transcribe_whisper.py", "-i", str(audio_file), "-o", "transcripts_tmp", "-m", "tiny", "-f", "srt,txt", "--device", "auto"])
        print(out)
        if code != 0:
            bad("Step #2 failed")
        ok("Step #2 completed")

        # Clean temp outputs
        shutil.rmtree(audio_dir, ignore_errors=True)
        shutil.rmtree(repo / "transcripts_tmp", ignore_errors=True)

    ok("Environment smoke test passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
