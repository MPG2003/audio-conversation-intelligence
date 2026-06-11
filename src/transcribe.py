"""
Batch audio transcription helper using OpenAI Whisper.

Usage (from project root):
    python -m src.transcribe --audio-dir audio --output data/raw/transcripts.csv

The script:
  - loads a Whisper model (default: small, CPU if CUDA is unavailable),
  - transcribes all audio files in the given folder,
  - appends results to a CSV (creates it with a header if missing),
  - skips files already present in the CSV unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Iterable, List, Set

import whisper


def list_audio_files(folder: Path) -> List[Path]:
    """Return sorted audio files with common extensions."""
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])


def load_seen_files(csv_path: Path) -> Set[str]:
    """Read existing CSV to avoid duplicate transcriptions."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["file_name"] for row in reader if "file_name" in row}


def write_header_if_needed(csv_path: Path, fieldnames: Iterable[str]) -> None:
    """Create CSV with header when file is missing or empty."""
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def choose_device(preferred: str | None) -> str:
    """Select CUDA if available unless user forces a device."""
    if preferred:
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def transcribe_file(model, audio_path: Path, language: str | None) -> dict:
    """Run Whisper transcription and return a compact record."""
    result = model.transcribe(
        str(audio_path),
        language=language,
        fp16=getattr(model, "device", None) and model.device.type == "cuda",
    )

    segments = result.get("segments") or []
    duration = segments[-1]["end"] if segments else None

    return {
        "file_name": audio_path.name,
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "duration_s": duration,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe audio with Whisper")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "audio",
        help="Folder containing audio files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "transcripts.csv",
        help="CSV path to append transcripts",
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large, etc.)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force device; defaults to CUDA if available else CPU",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code hint (e.g., 'en', 'hi'); autodetect if omitted",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output CSV instead of appending/skipping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_dir: Path = args.audio_dir
    output_csv: Path = args.output
    device = choose_device(args.device)

    audio_files = list_audio_files(audio_dir)
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_name", "text", "language", "duration_s", "timestamp"]
    write_header_if_needed(output_csv, fields)

    seen = set() if args.overwrite else load_seen_files(output_csv)
    model = whisper.load_model(args.model_size, device=device)

    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        for audio_path in audio_files:
            if audio_path.name in seen:
                # Skip already processed files unless --overwrite
                print(f"Skipping {audio_path.name} (already in CSV)")
                continue

            print(
                f"Transcribing {audio_path.name} with model='{args.model_size}' on {device}..."
            )
            record = transcribe_file(model, audio_path, args.language)
            writer.writerow(record)
            f.flush()

    print(f"Done. Results written to {output_csv}")


if __name__ == "__main__":
    main()
