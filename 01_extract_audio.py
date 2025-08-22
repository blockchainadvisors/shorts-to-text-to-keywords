# 01_extract_audio.py
# Usage:
#   pip install av
#   python 01_extract_audio.py -i .            # all *.mp4 in current dir
#   python 01_extract_audio.py -i . -r         # recurse
#   python 01_extract_audio.py -i 12.mp4       # single file
#   python 01_extract_audio.py -i . --force    # overwrite outputs

import argparse, pathlib, sys
import av  # PyAV (FFmpeg bindings)

AUDIO_EXT_FOR_CODEC = {"aac": "m4a", "alac": "m4a"}  # fallback .mka for others

def pick_ext(codec_name: str) -> str:
    return AUDIO_EXT_FOR_CODEC.get((codec_name or "").lower(), "mka")

def add_stream_compat(oc: av.container.OutputContainer, src_stream: av.stream.Stream) -> av.stream.Stream:
    try:
        return oc.add_stream(src_stream)
    except TypeError:
        try:
            return oc.add_stream(src_stream.codec_context)
        except TypeError:
            name = getattr(src_stream.codec_context, "name", None) or getattr(getattr(src_stream, "codec", None), "name", None)
            if not name:
                raise
            return oc.add_stream(name)

def compute_out_dir(src: pathlib.Path, out_root: pathlib.Path, root: pathlib.Path, flat: bool) -> pathlib.Path:
    return out_root if flat else out_root / src.parent.relative_to(root)

def extract_one(mp4_path: pathlib.Path, out_root: pathlib.Path, root: pathlib.Path, force: bool, flat: bool) -> int:
    ic = av.open(str(mp4_path))
    a_streams = [s for s in ic.streams if s.type == "audio"]
    if not a_streams:
        print(f"[skip] no audio: {mp4_path.name}")
        ic.close()
        return 0

    out_dir = compute_out_dir(mp4_path, out_root, root, flat)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for idx, s in enumerate(a_streams):
        codec_name = getattr(s.codec_context, "name", None) or getattr(getattr(s, "codec", None), "name", "")
        ext = pick_ext(codec_name)
        out_name = f"{mp4_path.stem}_audio.{ext}" if len(a_streams) == 1 else f"{mp4_path.stem}_audio-{idx}.{ext}"
        out_path = out_dir / out_name
        if out_path.exists() and not force:
            print(f"[keep] {out_path.relative_to(out_root)} exists")
            continue
        oc = av.open(str(out_path), mode="w")
        os_ = add_stream_compat(oc, s)
        outputs.append((s, oc, os_, out_path))

    if not outputs:
        ic.close()
        return 0

    route = {s: (oc, os_, out_path) for (s, oc, os_, out_path) in outputs}
    wrote = set()

    for packet in ic.demux([s for (s, *_rest) in outputs]):
        if packet.dts is None:
            continue
        oc, os_, out_path = route[packet.stream]
        packet.stream = os_
        oc.mux(packet)
        wrote.add(out_path)

    for _, oc, _, out_path in outputs:
        oc.close()
        print(f"[ok] {mp4_path.name} -> {out_path.relative_to(out_root)}")
    ic.close()
    return len(wrote)

def collect_files(inp: pathlib.Path, recursive: bool):
    if inp.is_file() and inp.suffix.lower() == ".mp4":
        return inp.parent, [inp]
    if inp.is_dir():
        pat = "**/*.mp4" if recursive else "*.mp4"
        return inp, sorted([p for p in inp.glob(pat) if p.is_file()])
    raise SystemExit("Input must be an .mp4 file or a directory containing .mp4 files.")

def main():
    ap = argparse.ArgumentParser(description="Extract audio tracks from MP4 via PyAV (remux, no re-encode).")
    ap.add_argument("-i", "--input", required=True, help="Directory or single .mp4 file")
    ap.add_argument("-o", "--outdir", default="audio", help="Output root folder (default: ./audio)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse when input is a directory")
    ap.add_argument("--flat", action="store_true", help="Do not mirror subfolder structure under outdir")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    out_root = pathlib.Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    root, files = collect_files(inp, args.recursive)
    total = 0
    for p in files:
        try:
            total += extract_one(p, out_root, root, force=args.force, flat=args.flat)
        except Exception as e:
            print(f"[err] {p.name}: {e}", file=sys.stderr)
    print(f"Done. Created {total} file(s) in '{out_root}'.")
if __name__ == "__main__":
    main()