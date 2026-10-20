import json
import tempfile
import threading
import unittest
from pathlib import Path

from toolshift.io import write_json
from toolshift.server import demo_run, PAGE
from toolshift.report import render_report


class DeliveryTests(unittest.TestCase):
    def test_atomic_writers_do_not_share_temporary_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            failures = []
            def writer(i):
                try:
                    write_json(path, {"run": i})
                except Exception as exc:
                    failures.append(exc)
            threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertFalse(failures)
            self.assertIn(json.loads(path.read_text())["run"], range(10))

    def test_demo_uses_real_state_and_rejects_invalid_inputs(self):
        success = demo_run({"domain": "travel", "suite": "mixed", "variant": 7, "policy": "static"})
        failed = demo_run({"domain": "travel", "suite": "mixed", "variant": 7, "policy": "legacy"})
        self.assertTrue(success["success"])
        self.assertTrue(success["robust_success"])
        self.assertFalse(failed["success"])
        self.assertTrue(any(step.get("observation") for step in success["trace"]))
        with self.assertRaises(ValueError):
            demo_run({"domain": "../../secret"})
        with self.assertRaises(ValueError):
            demo_run({"variant": -1})

    def test_report_escapes_policy_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = render_report([{"policy": "<script>alert(1)</script>",
                                   "summary": {"episodes": 0}}],
                                  Path(directory) / "report.html")
            content = path.read_text()
            self.assertNotIn("<script>alert(1)</script>", content)
            self.assertIn("&lt;script&gt;", content)


if __name__ == "__main__":
    unittest.main()
