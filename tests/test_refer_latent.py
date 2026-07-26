import unittest

import torch

from utils.refer_latent import compose_joint_refer_latent


class ReferLatentCompositionTest(unittest.TestCase):
    def test_composes_two_refers_over_historical_background(self):
        history = torch.full((1, 1, 2, 4, 8), 3.0)
        refers = [
            {"latent": torch.full((1, 1, 2, 4, 8), 1.0), "bbox": [0.0, 0.0, 0.5, 1.0]},
            {"latent": torch.full((1, 1, 2, 4, 8), 2.0), "bbox": [0.5, 0.0, 1.0, 1.0]},
        ]
        joint, used = compose_joint_refer_latent(
            refers, 2, 1, torch.float32, torch.device("cpu"),
            history_latent=history, margin=0.125,
        )
        self.assertEqual(joint.shape, (1, 2, 2, 4, 8))
        self.assertEqual(len(used), 2)
        self.assertTrue(torch.all(joint[:, :, :, :, 1:3] == 1.0))
        self.assertTrue(torch.all(joint[:, :, :, :, 5:7] == 2.0))
        self.assertTrue(torch.all(joint[:, :, :, :, :1] == 3.0))
        self.assertTrue(torch.all(joint[:, :, :, :, 3:5] == 3.0))

    def test_auto_layout_separates_references(self):
        refers = [
            {"latent": torch.ones((1, 1, 1, 2, 4))},
            {"latent": torch.full((1, 1, 1, 2, 4), 2.0)},
        ]
        joint, _ = compose_joint_refer_latent(
            refers, 1, 1, torch.float32, torch.device("cpu"), margin=0.0,
        )
        self.assertTrue(torch.all(joint[..., :2] == 1.0))
        self.assertTrue(torch.all(joint[..., 2:] == 2.0))


if __name__ == "__main__":
    unittest.main()
