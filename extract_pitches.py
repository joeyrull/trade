#!/usr/bin/env python3
"""
extract_pitches.py — Extract individual pitch clips from a long video.

Usage:
    python3 extract_pitches.py <video_file> [options]

Requires:
    ANTHROPIC_API_KEY environment variable

Steps:
    1. Transcribes the video with OpenAI Whisper (local, no API cost)
    2. Sends the transcript to Claude to identify each pitch boundary
    3. Cuts individual clips with ffmpeg
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path


def transcribe_video(video_path: str, model_size: str) -> dict:
    import whisper

    print(f"[1/3] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"[1/3] Transcribing '{video_path}' (this may take a few minutes)...")
    result = model.transcribe(video_path, verbose=False)
    return result


def detect_pitches(transcript_data: dict) -> list[dict]:
    import anthropic

    segments = transcript_data.get("segments", [])
    lines = []
    for seg in segments:
        h, rem = divmod(int(seg["start"]), 3600)
        m, s = divmod(rem, 60)
        lines.append(f"[{h:02d}:{m:02d}:{s:02d}] {seg['text'].strip()}")
    transcript_text = "\n".join(lines)

    client = anthropic.Anthropic()

    print("[2/3] Asking Claude to identify pitch boundaries...")
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are analyzing a transcript of a video containing multiple back-to-back business pitches.\n\n"
                    "Identify where each pitch begins and ends. Look for:\n"
                    "- A new team/company introducing themselves\n"
                    "- A moderator introducing the next presenter\n"
                    "- Natural hard transitions between distinct presentations\n\n"
                    "Return ONLY a JSON array — no explanation, no markdown fences — in this exact shape:\n"
                    "[\n"
                    '  {"pitch_number": 1, "start_time": 0.0, "end_time": 183.5, "title": "Acme Corp"},\n'
                    "  ...\n"
                    "]\n\n"
                    "Rules:\n"
                    "- start_time and end_time are floats in seconds\n"
                    "- title is the company/team name; use 'Pitch N' if unknown\n"
                    "- Cover the entire video — no gaps, no overlaps\n\n"
                    f"TRANSCRIPT:\n{transcript_text}"
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rstrip("`").strip()

    return json.loads(raw)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_clip(video_path: str, start: float, end: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", format_time(start),
        "-to", format_time(end),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ffmpeg error:\n{result.stderr[-800:]}", file=sys.stderr)
    return result.returncode == 0


def safe_filename(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "" for c in title).strip().replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual pitch clips from a long video using AI."
    )
    parser.add_argument("video", help="Path to the source video file")
    parser.add_argument(
        "--output-dir", "-o", default="clips",
        help="Directory to write clips into (default: clips/)",
    )
    parser.add_argument(
        "--whisper-model", "-m", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use for transcription (default: base; larger = more accurate but slower)",
    )
    parser.add_argument(
        "--transcript", "-t",
        help="Path to an existing transcript.json to skip re-transcription",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    video_path = args.video
    if not Path(video_path).exists():
        sys.exit(f"Error: video file not found: {video_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — transcribe
    if args.transcript:
        print(f"[1/3] Loading existing transcript from {args.transcript}...")
        with open(args.transcript) as f:
            transcript_data = json.load(f)
    else:
        transcript_data = transcribe_video(video_path, args.whisper_model)
        saved = out_dir / "transcript.json"
        with open(saved, "w") as f:
            json.dump(transcript_data, f, indent=2)
        print(f"      Transcript saved → {saved}")

    # Step 2 — detect pitches
    pitches = detect_pitches(transcript_data)

    print(f"\n[2/3] Detected {len(pitches)} pitch(es):")
    for p in pitches:
        print(
            f"      #{p['pitch_number']:02d}  {p['title']:<30}  "
            f"{format_time(p['start_time'])} → {format_time(p['end_time'])}"
        )

    manifest_path = out_dir / "pitches.json"
    with open(manifest_path, "w") as f:
        json.dump(pitches, f, indent=2)
    print(f"      Manifest saved → {manifest_path}\n")

    # Step 3 — extract clips
    print(f"[3/3] Extracting {len(pitches)} clip(s)...")
    ok = 0
    for p in pitches:
        fname = f"pitch_{p['pitch_number']:02d}_{safe_filename(p['title'])}.mp4"
        out_path = str(out_dir / fname)
        print(f"      {fname} ...", end="", flush=True)
        if extract_clip(video_path, p["start_time"], p["end_time"], out_path):
            size_mb = Path(out_path).stat().st_size / 1_048_576
            print(f" done ({size_mb:.1f} MB)")
            ok += 1
        else:
            print(" FAILED")

    print(f"\nFinished: {ok}/{len(pitches)} clips in '{out_dir}/'")
    if ok < len(pitches):
        sys.exit(1)


if __name__ == "__main__":
    main()
