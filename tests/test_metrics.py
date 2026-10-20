import unittest

from toolshift.metrics import paired_bootstrap, summarize


class MetricsTests(unittest.TestCase):
    def test_no_success_is_undefined_not_zero_cost(self):
        result = summarize([dict(episode_id="a", group_id="g", success=False,
                                 oracle_feasible=False, invariant_ok=True, query_cost=0,
                                 query_count=0, termination="abstained", action_count=0)])
        self.assertIsNone(result["query_cost_per_success"])
        self.assertIsNone(result["feasible_success_rate"])
        self.assertEqual(result["success_rate"], 0)

    def test_pairs_and_group_weights(self):
        left = [dict(episode_id=str(i), group_id="g" if i < 8 else "h", success=i < 8)
                for i in range(10)]
        right = [dict(row, success=False) for row in left]
        result = paired_bootstrap(left, right, resamples=100)
        self.assertEqual(result["difference_left_minus_right"], 0.5)
        self.assertEqual(result["independent_groups"], 2)
        with self.assertRaises(ValueError):
            paired_bootstrap(left, right[:-1])


if __name__ == "__main__":
    unittest.main()
