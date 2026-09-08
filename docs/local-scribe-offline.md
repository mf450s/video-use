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

The repository includes `scripts/verify_local_scribe.py` as a non-mocked smoke gate. It refuses cached transcript output, validates local ASR, AudioSet, and pyannote files, requires word timestamps, at least two speakers, and both `laughter` and `applause`, and can optionally run pack, subtitle generation, and preview render. Missing prerequisites return exit code `77` with an explicit `SKIP` message; add `--require` to turn that into a failure. Example:

```bash
rm -rf /tmp/video_use_real_smoke/edit
/home/charon/projects/video-use/.venv/bin/python \
  /home/charon/projects/video-use/scripts/verify_local_scribe.py \
  /tmp/video_use_real_smoke/input --edit-dir /tmp/video_use_real_smoke/edit \
  --model /home/charon/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66 \
  --event-model /home/charon/models/video-use/audioset-ast \
  --diarization-model /home/charon/models/video-use/pyannote-community-1 \
  --blackhole-proxy --require
```

Use a real fixture containing both event classes. Do not manufacture a laughter signal or use the degraded heuristic for this gate.

## verified strict offline E2E

The strict real-model run was verified on 2026-09-08 in `/home/charon/projects/video-use`, branch `feat/local-scribe-offline`. The run used these pre-provisioned local model paths:

| stage | verified path |
|---|---|
| ASR | `/home/charon/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` |
| audio events | `/home/charon/models/video-use/audioset-ast` |
| diarization | `/home/charon/models/video-use/pyannote-community-1` |

The runtime was isolated in `/home/charon/projects/video-use/.venv`. Verified versions were `faster-whisper 1.2.1`, `pyannote.audio 4.0.7`, `transformers 5.16.1`, `torch 2.14.0+cu130`, and `torchaudio 2.11.0+cu130`. The pyannote 4 `DiarizeOutput` compatibility path is covered by a regression test.

The real-model batch run used `/tmp/video_use_e2e_network_guard/input/` and `/tmp/video_use_e2e_network_guard/edit/`. It forced `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and blackhole proxy/endpoint values at `127.0.0.1:9`. The command completed with exit code `0` and processed two real MP4 inputs without network access or ElevenLabs calls.

The speech video produced `64` word entries with valid timestamps, `speaker_0` and `speaker_1`, `diarization: true`, and `backend: local`. A second real video input with the provisioned AudioSet model detected `applause` with score `0.8456`. `laughter` and `applause` are the only emitted event classes; a real laughter fixture was not available in this run and is therefore not claimed as observed.

Regression tests completed with exit code `0`: `28` tests passed. Pack completed with `2` transcripts and `6` phrases. Preview render completed with exit code `0`, produced `final.mp4`, `master.srt`, and loudness normalization. Timeline QC completed with exit code `0` and produced `timeline-qc.png`.

Verified artifacts are under `/tmp/video_use_e2e_network_guard/edit/`: `transcripts/real_clip.json`, `transcripts/event_only.json`, `takes_packed.md`, `master.srt`, `final.mp4`, `edl.json`, and `timeline-qc.png`.

### model provenance readback

The following revisions and SHA-256 values were read back from the exact files used by the run. The revision is the Hugging Face snapshot tree ID; it is not inferred from a model name.

| stage | repository | snapshot revision | runtime file | SHA-256 |
|---|---|---|---|---|
| ASR | `Systran/faster-whisper-base` | `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` | `model.bin` | `d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9` |
| audio events | `MIT/ast-finetuned-audioset-10-10-0.4593` | `f826b80d28226b62986cc218e5cec390b1096902` | `model.safetensors` | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` |
| diarization segmentation | `pyannote/speaker-diarization-community-1` | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | `segmentation/pytorch_model.bin` | `7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e` |
| diarization embedding | `pyannote/speaker-diarization-community-1` | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | `embedding/pytorch_model.bin` | `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929` |
| diarization PLDA | `pyannote/speaker-diarization-community-1` | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | `plda/plda.npz` | `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255` |
| diarization transform | `pyannote/speaker-diarization-community-1` | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | `plda/xvec_transform.npz` | `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f` |

For a fresh readback, run `sha256sum` against the runtime files listed above and compare every value. The AST directory also contains a duplicate `pytorch_model.bin`; the run loaded `model.safetensors`. Model configuration files are part of the same local directories and are not optional.

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
