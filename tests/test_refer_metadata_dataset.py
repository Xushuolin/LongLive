import json
import tempfile
import unittest
from pathlib import Path

from utils.dataset import MultiTextConcatDataset, eval_collate_fn


class ReferMetadataDatasetTest(unittest.TestCase):
    def test_refer_metadata_expands_with_shot_durations_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot_dir = root / "caption" / "sample"
            shot_dir.mkdir(parents=True)
            (shot_dir / "shot_durations.txt").write_text("2,2", encoding="utf-8")
            (shot_dir / "0.json").write_text(json.dumps({"caption": "first"}), encoding="utf-8")
            (shot_dir / "1.json").write_text(
                json.dumps({
                    "caption": "second",
                    "refers": [
                        {"image_path": "refs/subject.png", "role": "identity"},
                        {"image_path": "refs/prop.png", "role": "prop"},
                    ],
                }),
                encoding="utf-8",
            )

            item = MultiTextConcatDataset(str(root), num_blocks=4)[0]
            self.assertEqual(item["refers"][0], [])
            self.assertEqual(item["refers"][1], [])
            self.assertEqual(len(item["refers"][2]), 2)
            self.assertEqual(item["refers"][2][0]["role"], "identity")
            self.assertEqual(item["refers"][2][1]["role"], "prop")
            self.assertEqual(item["refers"][2][0]["image_path"], str(shot_dir / "refs" / "subject.png"))
            self.assertEqual(item["refers"][2][1]["image_path"], str(shot_dir / "refs" / "prop.png"))
            self.assertEqual(item["refers"][3][0]["image_path"], item["refers"][2][0]["image_path"])
            self.assertEqual(item["refers"][3][1]["image_path"], item["refers"][2][1]["image_path"])

            batch = eval_collate_fn([item])
            self.assertEqual(batch["refers"][0][2][0]["role"], "identity")
            self.assertEqual(batch["refers"][0][2][1]["role"], "prop")

    def test_continue_shot_does_not_insert_scene_cut_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_dir = root / "caption" / "sample"
            prompt_dir.mkdir(parents=True)
            (prompt_dir / "shot_durations.txt").write_text("2,3", encoding="utf-8")
            (prompt_dir / "001.json").write_text(
                json.dumps({"caption": "host waits"}), encoding="utf-8"
            )
            (prompt_dir / "002.json").write_text(
                json.dumps({
                    "caption": "host lifts bag",
                    "scene_cut": False,
                    "refers": [{"image_path": "bag.png", "role": "red handbag"}],
                }),
                encoding="utf-8",
            )

            item = MultiTextConcatDataset(str(root), num_blocks=5)[0]
            self.assertEqual(item["prompts"], [
                "host waits", "host waits", "host lifts bag", "host lifts bag", "host lifts bag",
            ])
            self.assertEqual(item["refers"][1], [])
            self.assertEqual(item["refers"][2][0]["role"], "red handbag")


if __name__ == "__main__":
    unittest.main()
