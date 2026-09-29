import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from examples.notebook_experiments import run_benchmark, summarize


class TestNotebookExperiments(unittest.TestCase):
    def test_run_benchmark_reads_json_and_builds_torchrun_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text(json.dumps([{"latency_median_ms": 1.5}]), encoding="utf-8")
            with mock.patch("examples.notebook_experiments.subprocess.run") as run:
                result = run_benchmark(
                    "/model",
                    tp_size=2,
                    collective_mode="buffered",
                    sampling_mode="distributed_packed",
                    output_path=output,
                )

            self.assertEqual(result["latency_median_ms"], 1.5)
            command = run.call_args.args[0]
            self.assertIn("torch.distributed.run", command)
            self.assertIn("--tp-collective-mode", command)
            self.assertIn("buffered", command)
            self.assertIn("distributed_packed", command)
            run.assert_called_once_with(mock.ANY, cwd=mock.ANY, check=True)

    def test_summarize_extracts_collective_metrics(self):
        rows = summarize(
            [
                {
                    "collective_mode": "legacy",
                    "sampling_mode": "distributed",
                    "tp_size": 2,
                    "latency_median_ms": 10,
                    "collective_stats": {
                        "all_gather": {"calls": 4, "bytes": 128},
                        "all_reduce": {"calls": 8},
                        "broadcast": {"calls": 2},
                    },
                }
            ]
        )
        self.assertEqual(rows[0]["all_gather_calls"], 4)
        self.assertEqual(rows[0]["all_gather_bytes"], 128)
        self.assertEqual(rows[0]["all_reduce_calls"], 8)
        self.assertEqual(rows[0]["broadcast_calls"], 2)


if __name__ == "__main__":
    unittest.main()