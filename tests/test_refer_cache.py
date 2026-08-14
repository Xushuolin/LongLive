import unittest

from utils.refer_cache import recent_cache_region


class ReferCacheRegionTest(unittest.TestCase):
    def test_selects_latest_requested_tokens(self):
        self.assertEqual(recent_cache_region(100, 20, 40), (60, 40))

    def test_never_overwrites_global_prefix(self):
        self.assertEqual(recent_cache_region(35, 20, 40), (20, 15))

    def test_empty_before_local_history_exists(self):
        self.assertEqual(recent_cache_region(20, 20, 40), (20, 0))


if __name__ == "__main__":
    unittest.main()
