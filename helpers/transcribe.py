"""Offline-first, Scribe-compatible video transcription.

The local backend uses an already-installed faster-whisper (preferred) or
openai-whisper model.  It never needs an API key.  Optional pyannote diarization
is used when a local pipeline is configured; otherwise speaker_0 is retained.
The emitted JSON deliberately keeps ElevenLabs' ``words`` list shape so the
packing and rendering helpers can consume either backend.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import requests

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def load_api_key() -> str:
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "ELEVENLABS_API_KEY" and value.strip().strip("\"'"):
                    return value.strip().strip("\"'")
    return os.environ.get("ELEVENLABS_API_KEY", "")


def count_audio_tracks(video_path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                          "-show_entries", "stream=index", "-of", "csv=p=0",
                          str(video_path)], capture_output=True, text=True, check=False)
    return len([line for line in out.stdout.splitlines() if line.strip()])


def peak_dbfs(wav_path: Path) -> float:
    peak = 0
    with wave.open(str(wav_path), "rb") as wav:
        while frames := wav.readframes(1 << 16):
            samples = array.array("h", frames)
            if samples:
                peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-map", f"0:a:{audio_track}",
                             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg could not extract audio track {audio_track + 1}: "
                           f"{result.stderr[-500:]}")


def call_scribe(audio_path: Path, api_key: str, language: str | None = None,
                num_speakers: int | None = None) -> dict:
    """Retained cloud adapter for compatibility; local is the default CLI mode."""
    if not api_key:
        raise RuntimeError("ElevenLabs backend selected but ELEVENLABS_API_KEY is missing")
    data: dict[str, str] = {"model_id": "scribe_v1", "diarize": "true",
                            "tag_audio_events": "true", "timestamps_granularity": "word"}
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)
    with open(audio_path, "rb") as audio:
        response = requests.post(SCRIBE_URL, headers={"xi-api-key": api_key},
                                 files={"file": (audio_path.name, audio, "audio/wav")},
                                 data=data, timeout=1800)
    if response.status_code != 200:
        raise RuntimeError(f"Scribe returned {response.status_code}: {response.text[:500]}")
    return response.json()


def _word(text: str, start: float, end: float, speaker: str | None = None) -> dict:
    item = {"type": "word", "text": text, "start": round(float(start), 3), "end": round(float(end), 3)}
    if speaker is not None:
        item["speaker_id"] = speaker
    return item


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _normalise_asr_segments(segments: Any) -> list[dict]:
    words: list[dict] = []
    for segment in segments:
        seg_words = _field(segment, "words") or []
        if seg_words:
            for item in seg_words:
                text = str(_field(item, "word", _field(item, "text", ""))).strip()
                start, end = _field(item, "start"), _field(item, "end")
                if text and start is not None and end is not None and float(end) >= float(start):
                    words.append(_word(text, start, end))
        else:
            text = str(_field(segment, "text", "")).strip()
            start, end = float(_field(segment, "start", 0.0)), float(_field(segment, "end", 0.0))
            tokens = text.split()
            if tokens and end >= start:
                step = (end - start) / len(tokens)
                words.extend(_word(token, start + i * step, start + (i + 1) * step)
                             for i, token in enumerate(tokens))
    return words


def _insert_spacing(words: list[dict]) -> list[dict]:
    result: list[dict] = []
    for index, item in enumerate(words):
        if index and item["start"] > result[-1].get("end", item["start"]):
            result.append({"type": "spacing", "text": " ", "start": result[-1]["end"],
                           "end": item["start"]})
        result.append(item)
    return result


def _load_samples(wav_path: Path) -> tuple[list[float], int]:
    import numpy as np
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def detect_audio_events(wav_path: Path) -> list[dict]:
    """Detect coarse local laughter/applause cues without a network model.

    This is intentionally conservative and deterministic: short energetic,
    noisy bursts become laughter and repeated bursts become applause.  A real
    event model can replace this function without changing the JSON contract.
    """
    import numpy as np
    samples, rate = _load_samples(wav_path)
    size, hop = max(1, int(rate * .20)), max(1, int(rate * .10))
    candidates: list[tuple[float, float, str]] = []
    for start in range(0, max(0, len(samples) - size + 1), hop):
        frame = samples[start:start + size]
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms < .12:
            continue
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))) if len(frame) > 1 else 0)
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        centroid = float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-9))
        # High ZCR/centroid is a useful laughter proxy; repeated broadband
        # transients are applause.  Do not emit events over very long regions.
        label = "laughter" if zcr > .16 and centroid > len(spectrum) * .12 else "applause"
        candidates.append((start / rate, min(len(samples) / rate, (start + size) / rate), label))
    merged: list[dict] = []
    for start, end, label in candidates:
        if merged and merged[-1]["text"] == label and start - merged[-1]["end"] <= .25:
            merged[-1]["end"] = round(end, 3)
        else:
            merged.append({"type": "audio_event", "text": label, "start": round(start, 3),
                           "end": round(end, 3)})
    return merged


def _speaker_segments(wav_path: Path, num_speakers: int | None) -> Any:
    """Return pyannote turns from a configured *local* pipeline, if possible."""
    model_path = os.environ.get("LOCAL_DIARIZATION_MODEL")
    if not model_path:
        return None
    if not Path(model_path).exists():
        print(f"warning: local diarization model path does not exist: {model_path}", file=sys.stderr)
        return None
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(model_path)
        diarization = pipeline(str(wav_path), num_speakers=num_speakers)
        return [(float(turn.start), float(turn.end), str(speaker))
                for turn, _, speaker in diarization.itertracks(yield_label=True)]
    except Exception as exc:
        print(f"warning: local diarization unavailable: {exc}", file=sys.stderr)
        return None


def transcribe_local(audio_path: Path, language: str | None = None,
                     num_speakers: int | None = None, model: str | None = None) -> dict:
    """Run local ASR and return the compatible Scribe response shape."""
    model_name = model or os.environ.get("LOCAL_ASR_MODEL")
    if not model_name:
        raise RuntimeError("local ASR model is not configured; install faster-whisper and set "
                           "LOCAL_ASR_MODEL to a downloaded model directory (or use --model)")
    if not Path(model_name).exists():
        raise RuntimeError(f"local ASR model path does not exist: {model_name}; provision model "
                           "weights first (offline mode never downloads them)")
    raw_segments = None
    errors: list[str] = []
    try:
        from faster_whisper import WhisperModel
        engine = WhisperModel(model_name, device=os.environ.get("LOCAL_ASR_DEVICE", "auto"),
                              compute_type=os.environ.get("LOCAL_ASR_COMPUTE_TYPE", "int8"),
                              local_files_only=True)
        raw_segments, info = engine.transcribe(str(audio_path), language=language,
                                               word_timestamps=True, vad_filter=True)
        raw_segments = list(raw_segments)
        detected_language = getattr(info, "language", language)
    except Exception as exc:
        errors.append(f"faster-whisper: {exc}")
        try:
            import whisper
            engine = whisper.load_model(model_name)
            result = engine.transcribe(str(audio_path), language=language, word_timestamps=True,
                                       fp16=False)
            raw_segments = result.get("segments", [])
            detected_language = result.get("language", language)
        except Exception as second:
            errors.append(f"openai-whisper: {second}")
            raise RuntimeError("no usable local ASR model; install faster-whisper or openai-whisper "
                               "and provide a downloaded model. " + " | ".join(errors)) from second

    words = _normalise_asr_segments(raw_segments)
    turns = _speaker_segments(audio_path, num_speakers)
    if turns:
        labels: dict[str, str] = {}
        for item in words:
            midpoint = (item["start"] + item["end"]) / 2
            for start, end, speaker in turns:
                if start <= midpoint <= end:
                    labels.setdefault(speaker, f"speaker_{len(labels)}")
                    item["speaker_id"] = labels[speaker]
                    break
            item.setdefault("speaker_id", "speaker_0")
    else:
        for item in words:
            item["speaker_id"] = "speaker_0"
    events = detect_audio_events(audio_path)
    combined = sorted(_insert_spacing(words) + events, key=lambda item: (item["start"],
                                                                          item["type"] != "audio_event"))
    return {"text": " ".join(item["text"] for item in words), "words": combined,
            "language_code": detected_language, "backend": "local",
            "diarization": bool(turns), "audio_events": True}


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def transcribe_one(video: Path, edit_dir: Path, api_key: str | None = None,
                   language: str | None = None, num_speakers: int | None = None,
                   verbose: bool = True, audio_track: int = 0, backend: str = "local",
                   model: str | None = None) -> Path:
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcript_path(edit_dir, video, audio_track)
    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path
    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)
    n_tracks = count_audio_tracks(video)
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)
        peak = peak_dbfs(audio)
        if peak < -60.0:
            hint = (f" try --audio-track " + " or ".join(str(i) for i in range(n_tracks) if i != audio_track)
                    if n_tracks > 1 else " check the source audio")
            raise RuntimeError(f"track {audio_track + 1} of {video.name} is silent ({peak:.1f} dBFS);" + hint)
        selected = backend
        if selected == "auto":
            selected = "elevenlabs" if (api_key or load_api_key()) else "local"
        if selected in ("elevenlabs", "scribe"):
            payload = call_scribe(audio, api_key or load_api_key(), language, num_speakers)
        elif selected == "local":
            payload = transcribe_local(audio, language, num_speakers, model)
        else:
            raise ValueError(f"unknown transcription backend: {backend!r} (use local, auto, or elevenlabs)")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"  saved: {out_path.name} ({len(payload.get('words', []))} timeline entries)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally (Scribe-compatible JSON)")
    ap.add_argument("video", type=Path)
    ap.add_argument("--edit-dir", type=Path, default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--audio-track", type=int, default=0)
    ap.add_argument("--backend", choices=("local", "auto", "elevenlabs", "scribe"), default="local")
    ap.add_argument("--offline", action="store_true", help="Alias for --backend local; never contacts ElevenLabs")
    ap.add_argument("--model", default=None, help="Downloaded local ASR model directory/name")
    args = ap.parse_args()
    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")
    backend = "local" if args.offline else args.backend
    transcribe_one(video, (args.edit_dir or video.parent / "edit").resolve(),
                   language=args.language, num_speakers=args.num_speakers,
                   audio_track=args.audio_track, backend=backend, model=args.model)


if __name__ == "__main__":
    main()
