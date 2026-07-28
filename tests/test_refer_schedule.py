import unittest

from utils.refer_schedule import progressive_refer_alpha


class ReferScheduleTest(unittest.TestCase):
    def test_linear_schedule_reaches_absolute_endpoints(self):
        values = [progressive_refer_alpha(i, 4, 0.1, 0.7, "linear") for i in range(4)]
        self.assertEqual(values, [0.1, 0.3, 0.5, 0.7])

    def test_cosine_schedule_is_monotonic(self):
        values = [progressive_refer_alpha(i, 5, 0.0, 1.0, "cosine") for i in range(5)]
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 1.0)
        self.assertEqual(values, sorted(values))

    def test_single_step_finishes_schedule(self):
        self.assertEqual(progressive_refer_alpha(0, 1, 0.1, 0.8, "linear"), 0.8)


if __name__ == "__main__":
    unittest.main()
