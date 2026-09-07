import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("video_use_transcribe", MODULE_PATH)
assert SPEC and SPEC.loader
transcribe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcribe)


class FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class FakeSegment:
    def __init__(self, text, start, end, words):
        self.text, self.start, self.end, self.words = text, start, end, words


class OfflineTranscriptTests(unittest.TestCase):
    def test_normalises_word_timestamps_and_preserves_gaps(self):
        words = transcribe._normalise_asr_segments([
            FakeSegment("hello world", 0, 1, [FakeWord("hello", .1, .4), FakeWord("world", .6, .9)])
        ])
        result = transcribe._insert_spacing(words)
        self.assertEqual(result[1]["type"], "spacing")
        self.assertEqual((result[1]["start"], result[1]["end"]), (.4, .6))
        self.assertEqual([x["text"] for x in result if x["type"] == "word"], ["hello", "world"])

    def test_local_backend_shape_with_injected_whisper(self):
        class FakeInfo:
            language = "en"

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return iter([FakeSegment("hello world", 0, 1,
                                         [FakeWord("hello", .1, .4), FakeWord("world", .6, .9)])]), FakeInfo()

        fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
        with tempfile.TemporaryDirectory() as model_dir, \
             patch.dict(sys.modules, {"faster_whisper": fake_module}), \
             patch.object(transcribe, "detect_audio_events", return_value=[]), \
             patch.object(transcribe, "_speaker_segments", return_value=None):
            result = transcribe.transcribe_local(Path("fixture.wav"), model=model_dir)
        self.assertEqual(result["backend"], "local")
        self.assertEqual(result["text"], "hello world")
        self.assertEqual([x["type"] for x in result["words"]], ["word", "spacing", "word"])
        self.assertEqual(result["words"][0]["speaker_id"], "speaker_0")

    def test_local_transcribe_one_never_reads_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"fixture")
            with patch.object(transcribe, "count_audio_tracks", return_value=1), \
                 patch.object(transcribe, "extract_audio"), \
                 patch.object(transcribe, "peak_dbfs", return_value=-10.0), \
                 patch.object(transcribe, "transcribe_local", return_value={"words": []}), \
                 patch.object(transcribe, "load_api_key", side_effect=AssertionError("API key read")):
                result = transcribe.transcribe_one(video, root / "edit", backend="local")
            self.assertTrue(result.exists())
            self.assertEqual(result.parent.name, "transcripts")

    def test_local_model_failure_is_clear(self):
        with self.assertRaisesRegex(RuntimeError, "model path does not exist"):
            transcribe.transcribe_local(Path("fixture.wav"), model="missing")

    def test_audio_event_entries_are_not_words(self):
        with patch.object(transcribe, "_load_samples", return_value=(__import__("numpy").ones(16000), 16000)):
            events = transcribe.detect_audio_events(Path("fixture.wav"))
        self.assertTrue(events)
        self.assertTrue(all(event["type"] == "audio_event" for event in events))
        self.assertIn(events[0]["text"], {"laughter", "applause"})


if __name__ == "__main__":
    unittest.main()
