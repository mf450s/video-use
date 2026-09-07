"""Batch transcription with the local/offline backend by default."""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import load_api_key, transcribe_one, transcript_path

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTS)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch transcription (local/offline by default)")
    ap.add_argument("videos_dir", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel workers; use 1 for local models to avoid duplicate model memory")
    ap.add_argument("--language", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--audio-track", type=int, default=0)
    ap.add_argument("--backend", choices=("local", "auto", "elevenlabs", "scribe"), default="local")
    ap.add_argument("--offline", action="store_true", help="Force local backend and never use network")
    ap.add_argument("--model", default=None, help="Downloaded local ASR model directory/name")
    args = ap.parse_args()
    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")
    edit_dir = (args.edit_dir or videos_dir / "edit").resolve()
    (edit_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")
    already_cached = [v for v in videos if transcript_path(edit_dir, v, args.audio_track).exists()]
    pending = [v for v in videos if v not in already_cached]
    print(f"found {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return
    backend = "local" if args.offline else args.backend
    api_key = load_api_key() if backend in ("auto", "elevenlabs", "scribe") else None
    if backend == "local" and args.workers > 1:
        print("note: local model will be loaded once per worker; --workers 1 is recommended")
    t0 = time.time()
    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transcribe_one, video=v, edit_dir=edit_dir, api_key=api_key,
                               language=args.language, num_speakers=args.num_speakers,
                               verbose=False, audio_track=args.audio_track, backend=backend,
                               model=args.model): v for v in pending}
        for future in as_completed(futures):
            video = futures[future]
            try:
                out = future.result()
                print(f"  + {video.stem}  →  {out.name}")
            except Exception as exc:
                errors.append((video, str(exc)))
                print(f"  x {video.stem}  FAILED: {exc}")
    print(f"\ndone in {time.time() - t0:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for video, message in errors:
            print(f"  {video.name}: {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
