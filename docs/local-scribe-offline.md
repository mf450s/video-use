# local scribe compatible pipeline

The local backend is strict by default. It does not silently replace diarization with `speaker_0` and it does not use an RMS or ZCR event heuristic.

## model contract

All three model paths must already exist before an offline run.

| stage | model | runtime format | path | requirements |
|---|---|---|---|---|
| ASR | `Systran/faster-whisper-base` | CTranslate2 `model.bin` plus `config.json`, tokenizer files | `LOCAL_ASR_MODEL` | CPU `int8` works; CUDA needs a compatible CUDA runtime and `LOCAL_ASR_DEVICE=cuda` |
| audio events | `MIT/ast-finetuned-audioset-10-10-0.4593` | Transformers AST PyTorch or safetensors checkpoint plus processor/config files | `LOCAL_EVENT_MODEL` | CPU works but is slower; CUDA uses `LOCAL_EVENT_DEVICE=cuda` |
| diarization | `pyannote/speaker-diarization-community-1` or a compatible pinned pyannote pipeline | local pipeline YAML and every referenced model/config/weight | `LOCAL_DIARIZATION_MODEL` | CPU works; CUDA is recommended for long recordings; provisioning may require Hugging Face access approval |

The repository does not contain model weights. Keep them outside the Git checkout. Record the exact Hugging Face revision and a checksum in the local provisioning manifest, not in Git.

The default event threshold is `0.35`. Override it with `LOCAL_EVENT_THRESHOLD`. The classifier is asked for local files only and runs on overlapping ten second windows. Only labels matching `laughter` or `applause` are emitted.

## one time provisioning

Provision with network access on the same host, then run with network access disabled. The commands below create ordinary directories that can be mounted or copied to an offline host.

```bash
mkdir -p /mnt/nas/models/video-use/{faster-whisper-base,audioset-ast,pyannote-community-1}

# ASR. This is the CTranslate2 layout expected by faster-whisper.
HF_HOME=/mnt/nas/models/huggingface \
  huggingface-cli download Systran/faster-whisper-base \
  --local-dir /mnt/nas/models/video-use/faster-whisper-base

# Audio events. Keep config, preprocessor, label map, and weights together.
HF_HOME=/mnt/nas/models/huggingface \
  huggingface-cli download MIT/ast-finetuned-audioset-10-10-0.4593 \
  --local-dir /mnt/nas/models/video-use/audioset-ast

# Diarization. Accept the model terms and provide a temporary HF token only
# for this provisioning command. Never put the token in a file or command log.
HF_HOME=/mnt/nas/models/huggingface \
  huggingface-cli download pyannote/speaker-diarization-community-1 \
  --local-dir /mnt/nas/models/video-use/pyannote-community-1
```

For a gated diarization model, the one-time command fails until the account has accepted the model terms. The runtime does not need the token after all referenced files are local.

Verify that provisioning is complete before disconnecting the network:

```bash
test -f /mnt/nas/models/video-use/faster-whisper-base/model.bin
test -f /mnt/nas/models/video-use/audioset-ast/config.json
test -f /mnt/nas/models/video-use/audioset-ast/preprocessor_config.json
test -f /mnt/nas/models/video-use/pyannote-community-1/config.yaml
```

Also save the exact revisions and checksums:

```bash
sha256sum /mnt/nas/models/video-use/faster-whisper-base/model.bin \
  /mnt/nas/models/video-use/audioset-ast/*.safetensors \
  /mnt/nas/models/video-use/audioset-ast/*.bin
```

## offline execution

```bash
export LOCAL_ASR_MODEL=/mnt/nas/models/video-use/faster-whisper-base
export LOCAL_EVENT_MODEL=/mnt/nas/models/video-use/audioset-ast
export LOCAL_DIARIZATION_MODEL=/mnt/nas/models/video-use/pyannote-community-1
export HF_HUB_OFFLINE=1

python helpers/transcribe.py /path/to/take.mp4 --offline \
  --model "$LOCAL_ASR_MODEL" \
  --event-model "$LOCAL_EVENT_MODEL" \
  --diarization-model "$LOCAL_DIARIZATION_MODEL"
```

The local path does not call `load_api_key`, `requests.post`, or ElevenLabs. `--backend elevenlabs` remains an explicit separate path.

A degraded run is available only when the operator knowingly accepts its limits:

```bash
python helpers/transcribe.py /path/to/take.mp4 --offline --degraded-mode \
  --model "$LOCAL_ASR_MODEL"
```

That mode marks every word `speaker_0` and uses the legacy signal heuristic. It is not a full local Scribe replacement and must not be used as the production E2E proof.

## verification sequence

Run these in the footage directory. All generated files stay below `edit/`.

```bash
python helpers/transcribe_batch.py /path/to/takes --offline --workers 1 \
  --model "$LOCAL_ASR_MODEL" --event-model "$LOCAL_EVENT_MODEL" \
  --diarization-model "$LOCAL_DIARIZATION_MODEL"
python helpers/pack_transcripts.py --edit-dir /path/to/takes/edit
python helpers/render.py /path/to/takes/edit/edl.json \
  -o /path/to/takes/edit/final.mp4 --build-subtitles --preview
python helpers/timeline_view.py /path/to/takes/edit/final.mp4 0 5
```

Inspect the transcript JSON for:

- word entries with timestamps and at least two speaker IDs when the source has speaker changes;
- `audio_event` entries labelled `laughter` or `applause` with scores;
- `diarization: true` and the model paths;
- no API calls in the offline network trace.

Then inspect `takes_packed.md`, `master.srt`, the preview or final render, and the timeline QC PNG. A passing unit test with injected models is not a real-model E2E result.

## resource and cost notes

- Local ASR, event classification, diarization, packing, subtitles, render, and QC consume no ElevenLabs or LLM API tokens.
- Provisioning downloads model weights and consumes disk and bandwidth, but no inference token quota.
- `--backend elevenlabs` can create ElevenLabs usage and remains opt-in.
- CPU inference time depends on recording length and model size. AST and pyannote use materially more RAM than the tiny ASR model. Measure on the target host before increasing batch workers.
- Do not run multiple local workers unless the host has enough RAM for one copy of every loaded model per worker.

## failure behavior

- missing ASR path: clear error before inference;
- missing event model: clear error, never heuristic fallback;
- missing or invalid diarization path: clear error, never silent `speaker_0` fallback;
- broken local pipeline: clear error with `--degraded-mode` required to continue;
- no network is required after provisioning. `HF_HUB_OFFLINE=1` and local model paths make accidental downloads fail closed.
