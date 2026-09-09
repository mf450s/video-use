#!/usr/bin/env python3
"""Run and validate a real local Scribe-compatible transcription smoke test.

This script deliberately has no mocks.  Missing models or footage return the
POSIX skip code 77 unless ``--require`` is supplied, in which case they fail.
Use a fresh edit directory so a cached JSON cannot masquerade as inference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SKIP = 77
VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}
REQUIRED_EVENTS = {"laughter", "applause"}


def missing_prerequisites(args: argparse.Namespace, videos: list[Path]) -> list[str]:
    missing: list[str] = []
    for label, value in (
        ("ASR model", args.model),
        ("audio-event model", args.event_model),
        ("diarization model", args.diarization_model),
    ):
        if not value:
            missing.append(f"{label}: argument or corresponding LOCAL_*_MODEL variable is missing")
        elif not Path(value).is_dir():
            missing.append(f"{label}: directory does not exist: {value}")
    if args.model and not (Path(args.model) / "model.bin").exists():
        missing.append(f"ASR model: model.bin missing below {args.model}")
    if args.event_model and not (Path(args.event_model) / "config.json").exists():
        missing.append(f"audio-event model: config.json missing below {args.event_model}")
    if args.event_model and not any(
        (Path(args.event_model) / name).exists() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        missing.append(f"audio-event model: no PyTorch or safetensors weights below {args.event_model}")
    if args.diarization_model:
        diar = Path(args.diarization_model)
        for relative in ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin"):
            if not (diar / relative).exists():
                missing.append(f"diarization model: {relative} missing below {args.diarization_model}")
    if not videos:
        missing.append(f"input directory contains no supported video files: {args.input_dir}")
    return missing


def run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}")


def validate_transcripts(edit_dir: Path, videos: list[Path], min_speakers: int,
                         required_events: set[str], audio_track: int) -> dict[str, Any]:
    transcript_dir = edit_dir / "transcripts"
    files: list[Path] = []
    for video in videos:
        suffix = "" if audio_track == 0 else f".track{audio_track}"
        path = transcript_dir / f"{video.stem}{suffix}.json"
        if not path.is_file():
            raise RuntimeError(f"missing transcript output: {path}")
        files.append(path)

    all_speakers: set[str] = set()
    all_events: set[str] = set()
    word_count = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("backend") != "local":
            raise RuntimeError(f"{path}: backend is not local: {data.get('backend')!r}")
        if data.get("diarization") is not True:
            raise RuntimeError(f"{path}: diarization is not true")
        words = data.get("words")
        if not isinstance(words, list):
            raise RuntimeError(f"{path}: words is not a list")
        for item in words:
            if item.get("type") == "word":
                word_count += 1
                start, end = item.get("start"), item.get("end")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
                    raise RuntimeError(f"{path}: invalid word timestamp: {item}")
                speaker = item.get("speaker_id")
                if not speaker:
                    raise RuntimeError(f"{path}: word has no speaker_id: {item}")
                all_speakers.add(str(speaker))
            elif item.get("type") == "audio_event":
                label = item.get("text")
                if label in REQUIRED_EVENTS:
                    all_events.add(label)
                start, end = item.get("start"), item.get("end")
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end < start:
                    raise RuntimeError(f"{path}: invalid event timestamp: {item}")
    if word_count == 0:
        raise RuntimeError("real local run produced no word entries")
    if len(all_speakers) < min_speakers:
        raise RuntimeError(f"expected at least {min_speakers} speakers, observed {sorted(all_speakers)}")
    absent = required_events - all_events
    if absent:
        raise RuntimeError(f"required real AudioSet events absent: {sorted(absent)}; observed {sorted(all_events)}")
    return {"transcripts": [str(path) for path in files], "word_count": word_count,
            "speakers": sorted(all_speakers), "events": sorted(all_events)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a non-mocked local video-use transcription run")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--edit-dir", type=Path, default=None)
    parser.add_argument("--model", default=os.environ.get("LOCAL_ASR_MODEL"))
    parser.add_argument("--event-model", default=os.environ.get("LOCAL_EVENT_MODEL"))
    parser.add_argument("--diarization-model", default=os.environ.get("LOCAL_DIARIZATION_MODEL"))
    parser.add_argument("--audio-track", type=int, default=0)
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--require-event", action="append", dest="required_events", default=None,
                        help="Required event label; defaults to laughter and applause")
    parser.add_argument("--edl", type=Path, default=None,
                        help="Also run pack, subtitle generation, preview render, and validate outputs")
    parser.add_argument("--blackhole-proxy", action="store_true",
                        help="Route HTTP(S) proxies to 127.0.0.1:9 in addition to offline flags")
    parser.add_argument("--require", action="store_true", help="Turn missing prerequisites into failure instead of skip")
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    if not args.input_dir.is_dir():
        print(f"input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1
    args.edit_dir = (args.edit_dir or args.input_dir / "edit").resolve()
    videos = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix in VIDEO_EXTS)
    missing = missing_prerequisites(args, videos)
    if missing:
        print("SKIP: prerequisites are not complete:")
        for item in missing:
            print(f"  - {item}")
        return 1 if args.require else SKIP

    args.edit_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir = args.edit_dir / "transcripts"
    if transcript_dir.exists() and any(transcript_dir.glob("*.json")):
        print(f"refusing cached-output validation; use a fresh edit directory: {transcript_dir}", file=sys.stderr)
        return 1

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("ELEVENLABS_API_KEY",):
        env.pop(key, None)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if args.blackhole_proxy:
        env.update({"HF_ENDPOINT": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                    "HTTPS_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9"})

    try:
        run([sys.executable, str(repo / "helpers" / "transcribe_batch.py"), str(args.input_dir),
             "--offline", "--workers", "1", "--audio-track", str(args.audio_track),
             "--edit-dir", str(args.edit_dir), "--model", args.model,
             "--event-model", args.event_model, "--diarization-model", args.diarization_model], env, repo)
        result = validate_transcripts(args.edit_dir, videos, args.min_speakers,
                                      set(args.required_events or REQUIRED_EVENTS), args.audio_track)
        if args.edl:
            edl = args.edl.resolve()
            if not edl.is_file():
                raise RuntimeError(f"EDL does not exist: {edl}")
            run([sys.executable, str(repo / "helpers" / "pack_transcripts.py"),
                 "--edit-dir", str(args.edit_dir)], env, repo)
            final = args.edit_dir / "final.mp4"
            run([sys.executable, str(repo / "helpers" / "render.py"), str(edl), "-o", str(final),
                 "--build-subtitles", "--preview"], env, repo)
            for path in (args.edit_dir / "takes_packed.md", args.edit_dir / "master.srt", final):
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"pipeline output missing or empty: {path}")
            result["pipeline_outputs"] = [str(args.edit_dir / "takes_packed.md"),
                                          str(args.edit_dir / "master.srt"), str(final)]
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
