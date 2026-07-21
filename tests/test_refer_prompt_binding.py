import unittest

from utils.refer_prompt import (
    append_refer_binding_to_prompts,
    build_refer_binding_text,
    build_refer_kv_prompts,
)


class ReferPromptBindingTest(unittest.TestCase):
    def test_builds_role_layout_and_joint_scene_binding(self):
        refers = [
            {"role": "woman", "layout": "on the left"},
            {"role": "dog", "layout": "on the right", "description": "small white dog"},
        ]

        binding = build_refer_binding_text(refers, joint_scene=True)

        self.assertIn("Reference image 1 is the woman on the left", binding)
        self.assertIn("Reference image 2 is the dog, small white dog on the right", binding)
        self.assertIn("one coherent multi-object visual anchor", binding)
        self.assertIn("Keep the references distinct", binding)

    def test_rewrites_only_chunks_with_refers(self):
        prompts = ["first shot", "second shot"]
        refers = [[], [{"role": "robot", "layout": "center"}]]

        rewritten = append_refer_binding_to_prompts(prompts, refers)
        kv_prompts = build_refer_kv_prompts(prompts, refers)

        self.assertEqual(rewritten[0], "first shot")
        self.assertIn("Reference image 1 is the robot center", rewritten[1])
        self.assertEqual(kv_prompts[0], "first shot")
        self.assertIn("Reference image 1 is the robot center", kv_prompts[1])


if __name__ == "__main__":
    unittest.main()
