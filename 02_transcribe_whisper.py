#!/usr/bin/env python3
r"""
transcribe_whisper.py — Batch transcribe/translate audio/video to SRT/VTT/TXT using openai-whisper.

Prerequisites:
--------
pip install av openai-whisper torch

Overview
--------
- Accepts a single file or a directory (optional recursion).
- Mirrors the input folder structure under the output directory (or flattens with --flat).
- Writes any combination of SRT, VTT, and TXT.
- Loads the Whisper model once for fast batch processing.

Requirements
------------
pip install openai-whisper torch
(Whisper requires FFmpeg installed and discoverable on PATH.)

Examples
--------
# Transcribe all audio/video in current folder -> transcripts/ (mirror subfolders)
python transcribe_whisper.py -i . -o transcripts

# Recurse subfolders under input
python transcribe_whisper.py -i . -r -o transcripts

# Single file
python transcribe_whisper.py -i .\audio\10_audio.m4a -o transcripts

# Flat output (no subfolders), force language to English, VTT+TXT only
python transcribe_whisper.py -i . -o transcripts --flat --language en -f vtt,txt

# Translate to English instead of transcribe
python transcribe_whisper.py -i . -o transcripts --task translate --language en

Flags
-----
-i/--input      : Path to file or directory.
-o/--outdir     : Output root (default: ./transcripts).
-r/--recursive  : Recurse when input is a directory.
--flat          : Do not mirror subfolder structure under outdir.
-m/--model      : Whisper model (tiny|base|small|medium|large). Default: small.
-f/--formats    : Comma list: srt,vtt,txt. Default: srt,vtt,txt.
--language      : Force language code (e.g., en, ro). Default: auto-detect.
--task          : transcribe | translate. Default: transcribe.
--device        : auto|cpu|cuda. Default: auto (cuda if available).
"""

import argparse, pathlib, sys, numpy as np
import torch, whisper, av

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mka", ".aac", ".wma", ".mp4", ".mov", ".mkv", ".webm"}

# ---------- Local SRT/VTT writers ----------
def _fmt_ts(sec: float, vtt: bool) -> str:
    sec = 0.0 if sec is None else max(0.0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60); ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}{'.' if vtt else ','}{ms:03d}"

def write_srt_from_segments(segments, file):
    for i, seg in enumerate(segments, start=1):
        file.write(f"{i}\n{_fmt_ts(seg.get('start'), False)} --> {_fmt_ts(seg.get('end'), False)}\n{(seg.get('text') or '').strip()}\n\n")

def write_vtt_from_segments(segments, file):
    file.write("WEBVTT\n\n")
    for seg in segments:
        file.write(f"{_fmt_ts(seg.get('start'), True)} --> {_fmt_ts(seg.get('end'), True)}\n{(seg.get('text') or '').strip()}\n\n")
# -------------------------------------------

def load_audio_pyav_to_tensor(path: pathlib.Path, target_rate=16000) -> torch.Tensor:
    """
    Decode with PyAV, resample to mono 16 kHz, return float32 tensor in [-1, 1].
    Never calls ffmpeg.exe.
    """
    with av.open(str(path)) as ic:
        astreams = [s for s in ic.streams if s.type == "audio"]
        if not astreams:
            raise RuntimeError("no audio stream")
        astream = astreams[0]

        resampler = av.audio.resampler.AudioResampler(
            format="s16",     # 16-bit signed
            layout="mono",    # force mono
            rate=target_rate  # 16 kHz
        )

        chunks = []
        # Demux → decode → resample (resample may yield 0..N frames per input frame)
        for packet in ic.demux(astream):
            for frame in packet.decode():
                out = resampler.resample(frame)
                if out is None:
                    continue
                if not isinstance(out, list):
                    out = [out]
                for oframe in out:
                    arr = oframe.to_ndarray()  # shape: (channels, samples) or (samples,)
                    if arr.ndim == 2:
                        # already mono per resampler, but be safe
                        arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
                    chunks.append(arr.astype(np.int16, copy=False))

    if not chunks:
        return torch.zeros(0, dtype=torch.float32)

    int16 = np.concatenate(chunks)
    return torch.from_numpy(int16.astype(np.float32) / 32768.0)

def iter_inputs(inp: pathlib.Path, recursive: bool):
    if inp.is_file():
        if inp.suffix.lower() in AUDIO_EXTS: yield inp.parent, inp
        else: raise SystemExit(f"Unsupported file type: {inp.suffix}")
    elif inp.is_dir():
        pat = "**/*" if recursive else "*"
        for p in sorted(inp.glob(pat)):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield inp, p
    else:
        raise SystemExit("Input must be a file or directory")

def out_base_for(file: pathlib.Path, input_root: pathlib.Path, out_root: pathlib.Path, flat: bool) -> pathlib.Path:
    if flat:
        out_root.mkdir(parents=True, exist_ok=True); return out_root / file.stem
    rel_dir = file.parent.relative_to(input_root); out_dir = out_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True); return out_dir / file.stem

def save_txt(text: str, out_path: pathlib.Path):
    out_path.write_text((text or "").strip() + "\n", encoding="utf-8")

def transcribe_tensor(model, audio: torch.Tensor, language: str|None, task: str, fp16: bool):
    # Feed tensor → NO ffmpeg call
    return model.transcribe(audio=audio, language=(language or None), task=task, fp16=fp16, verbose=False)

def main():
    ap = argparse.ArgumentParser(description="Batch transcribe/translate (no ffmpeg CLI) with Whisper + PyAV.")
    ap.add_argument("-i","--input", required=True, help="Directory or single audio/video file")
    ap.add_argument("-o","--outdir", default="transcripts", help="Output root (default: ./transcripts)")
    ap.add_argument("-r","--recursive", action="store_true", help="Recurse when input is a directory")
    ap.add_argument("--flat", action="store_true", help="Do not mirror subfolders under outdir")
    ap.add_argument("-m","--model", default="small", help="Whisper model: tiny|base|small|medium|large")
    ap.add_argument("-f","--formats", default="srt,vtt,txt", help="Comma list: srt,vtt,txt")
    ap.add_argument("--language", default=None, help="Force language code (e.g., en, ro). Default: auto-detect")
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"], help="Whisper task")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device")
    args = ap.parse_args()

    in_path = pathlib.Path(args.input); out_root = pathlib.Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    fp16 = (device == "cuda")
    print(f"[load] model={args.model} device={device} fp16={fp16}")
    model = whisper.load_model(args.model, device=device)

    write_formats = {x.strip().lower() for x in args.formats.split(",") if x.strip()}
    total = 0

    for root, file in iter_inputs(in_path, args.recursive):
        out_base = out_base_for(file, root, out_root, args.flat)
        try:
            audio = load_audio_pyav_to_tensor(file)
            if audio.numel() == 0:
                raise RuntimeError("empty audio")
            result = transcribe_tensor(model, audio, args.language, args.task, fp16)
            segs = result.get("segments", []) or []
            txt  = (result.get("text") or "").strip()

            if "txt" in write_formats:  save_txt(txt, out_base.with_suffix(".txt"))
            if "srt" in write_formats:  out_base.with_suffix(".srt").write_text(
                "".join(
                    f"{i}\n{_fmt_ts(s.get('start'), False)} --> {_fmt_ts(s.get('end'), False)}\n{(s.get('text') or '').strip()}\n\n"
                    for i, s in enumerate(segs, start=1)
                ), encoding="utf-8"
            )
            if "vtt" in write_formats:
                with out_base.with_suffix(".vtt").open("w", encoding="utf-8") as f:
                    write_vtt_from_segments(segs, f)

            print(f"[ok] {file} -> {out_base.parent}")
            total += 1
        except Exception as e:
            print(f"[err] {file.name}: {e}", file=sys.stderr)

    print(f"Done. Transcribed {total} file(s) to '{out_root}'.")

if __name__ == "__main__":
    main()