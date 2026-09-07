import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "helpers" / "pack_transcripts.py"
SPEC = importlib.util.spec_from_file_location("video_use_pack", MODULE_PATH)
assert SPEC and SPEC.loader
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


class PackCompatibilityTests(unittest.TestCase):
    def test_scribe_and_local_entries_pack_identically(self):
        words = [
            {"type": "word", "text": "hello", "start": 0.1, "end": 0.3, "speaker_id": "speaker_0"},
            {"type": "spacing", "text": " ", "start": 0.3, "end": 1.0},
            {"type": "word", "text": "world", "start": 1.1, "end": 1.4, "speaker_id": "speaker_0"},
            {"type": "audio_event", "text": "laughter", "start": 1.5, "end": 1.7, "speaker_id": "speaker_0"},
        ]
        phrases = pack.group_into_phrases(words)
        self.assertEqual(len(phrases), 2)
        self.assertEqual(phrases[0]["text"], "hello")
        self.assertEqual(phrases[1]["text"], "world (laughter)")
        self.assertEqual(phrases[1]["speaker_id"], "speaker_0")

    def test_json_fixture_is_consumed_by_pack_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.json"
            path.write_text(json.dumps({"text": "hi", "words": [
                {"type": "word", "text": "hi", "start": 0, "end": .2}
            ]}))
            name, duration, phrases = pack.pack_one_file(path, .5)
        self.assertEqual((name, duration), ("take", .2))
        self.assertEqual(phrases[0]["text"], "hi")


if __name__ == "__main__":
    unittest.main()
