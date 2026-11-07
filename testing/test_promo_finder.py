"""Run from project root: python -m unittest discover -s testing -v"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))
from promo_finder import (ts_to_s, s_to_ts, read_inputs, parse_episode, run_analysis,
                          score_candidate, spans_dialogue, selected_dialogue)


class Tests(unittest.TestCase):
    def test_timestamp_round_trip(self):
        for t in ("00:00.00", "01:02.75", "23:59.01"):
            self.assertEqual(s_to_ts(ts_to_s(t)), t)
        with self.assertRaises(ValueError):
            ts_to_s("00:63.00")

    def test_new_genre_two_episode_show(self):
        pack = ROOT / "demo_input" / "unseen_space_show"
        with tempfile.TemporaryDirectory() as td:
            p, eps, c = run_analysis(pack, td, count=2)
            self.assertEqual(p["processed_episode_count"], 2)
            self.assertEqual(len(p["selected_promos"]), 2)
            self.assertGreater(len(c), 10)
            for item in p["selected_promos"]:
                sg = item["segments"][0]
                self.assertGreaterEqual(item["duration_seconds"], 20)
                self.assertLessEqual(item["duration_seconds"], 60)
                self.assertLessEqual(item["first_beat_delay_seconds"], 3)
                self.assertFalse(spans_dialogue(next(e for e in eps if e.number == item["episode"]), ts_to_s(sg["start"])))
                self.assertFalse(spans_dialogue(next(e for e in eps if e.number == item["episode"]), ts_to_s(sg["end"])))
            self.assertTrue(Path(td, "editor_review_template.csv").exists())
            self.assertTrue(Path(td, "run_summary.json").exists())

    def test_rhetorical_question_and_incomplete_opening_warning(self):
        src = """EPISODE NUMBER: Episode 1
SCENE 01
Scene start: 00:00.00
Scene end: 00:45.00

**ALICE:**
Start: (00:00.00) | End: (00:03.00)
"While everyone waited."

**BOB:**
Start: (00:04.00) | End: (00:08.00)
"I was searching the room for the missing badge."

**ALICE:**
Start: (00:08.50) | End: (00:13.00)
"You said there was a camera in the corridor."

**BOB:**
Start: (00:13.50) | End: (00:18.00)
"Yes, but the recording has disappeared."

**ALICE:**
Start: (00:18.50) | End: (00:22.50)
"How dare you accuse me?"

**BOB:**
Start: (00:23.00) | End: (00:28.00)
"I think somebody else entered the building."

**ALICE:**
Start: (00:28.50) | End: (00:33.00)
"Then look for more evidence before deciding."
"""
        ep = parse_episode("episode-001.txt", src, 1)
        item = score_candidate(ep, 0, 22.5)
        self.assertIsNotNone(item)
        self.assertTrue(any("mid-thought" in w for w in item["quality_warnings"]))
        self.assertTrue(any("rhetorical" in w for w in item["quality_warnings"]))

    def test_rejects_untimed_or_invalid_episode(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "episode-001.txt"
            f.write_text("EPISODE NUMBER: Episode 1\nA story without timestamps", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Cannot parse enough"):
                run_analysis(f, Path(td) / "out")

    def test_rejects_duplicate_episode_ids(self):
        with tempfile.TemporaryDirectory() as td:
            zfile = Path(td) / "same.zip"
            demo = (ROOT / "demo_input" / "demo_episode_001.txt").read_text()
            with zipfile.ZipFile(zfile, "w") as z:
                z.writestr("episode-001-first.txt", demo)
                z.writestr("episode-001-second.txt", demo)
            with self.assertRaisesRegex(ValueError, "Duplicate episode"):
                run_analysis(zfile, Path(td) / "out")

    def test_json_output_and_csv_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            p, eps, c = run_analysis(ROOT / "demo_input" / "demo_episode_001.txt", td, count=2)
            readback = json.loads((Path(td) / "promos_ranked.json").read_text())
            self.assertEqual(readback["selected_promo_count"], p["selected_promo_count"])
            self.assertEqual(readback["run_mode"], "Offline NLP + show-independent transparent scoring + TF-IDF deduplication")
            self.assertTrue(all(i["score"] <= 100 for i in p["selected_promos"]))

    def test_no_show_specific_royal_themes_in_scoring(self):
        source = (ROOT / "prototype" / "promo_finder.py").read_text(encoding="utf-8").lower()
        # A generic model must not rely on fixed royal-genre query templates.
        for keyword in ("duke", "duchess", "royal family", "throne", "wedding drama"):
            self.assertNotIn(keyword, source)


if __name__ == "__main__":
    unittest.main()
