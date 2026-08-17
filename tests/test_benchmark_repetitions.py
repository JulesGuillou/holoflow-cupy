from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from holoflow_benchmarks.config import ExecutionMode, load_benchmark_config
from holoflow_benchmarks.nsys_profile import single_mode_config
from holoflow_benchmarks.reporting import format_report
from holoflow_benchmarks.stats import (
    BenchmarkSeries,
    BenchmarkStats,
    run_repeated_modes,
    summarize_throughput,
)


BASE_CONFIG = {
    "input": {"file_path": "synthetic.holo"},
    "signal": {
        "sample_rate_hz": 37_000.0,
        "doppler_low_hz": 8_000.0,
        "doppler_high_hz": 16_000.0,
        "propagation_distance_m": 0.486,
        "wavelength_m": 8.52e-7,
        "dx_m": 2.0e-5,
        "dy_m": 2.0e-5,
    },
    "schedule": {"batch_frames": 32, "batches_per_output": 64},
    "benchmark": {"seconds": 10.0, "warmup_outputs": 1},
    "display": {
        "show_image": False,
        "contrast_roi_radius": 0.8,
        "contrast_low_percentile": 0.2,
        "contrast_high_percentile": 99.8,
    },
    "dtypes": {
        "acquisition": "uint8",
        "real": "float32",
        "complex": "complex64",
    },
    "execution": {
        "modes": [
            {
                "name": "test-mode",
                "precompute_static_tensors": True,
                "preallocate_work_buffers": True,
            }
        ]
    },
}


def make_stats(mode_name: str, input_fps: float) -> BenchmarkStats:
    return BenchmarkStats(
        mode_name=mode_name,
        precompute_static_tensors=True,
        preallocate_work_buffers=True,
        seconds=10.0,
        frames=int(input_fps * 10),
        batches=100,
        outputs=10,
        input_fps=input_fps,
        batches_per_second=10.0,
        outputs_per_second=1.0,
        wall_ms_per_output=1000.0,
        h2d_gbps=1.0,
        cast_effective_gbps=4.0,
        d2h_output_mbps=2.0,
        shape=(16, 16),
        file_dtype="uint8",
        host_dtype="uint8",
        device_input_dtype="uint8",
        real_dtype="float32",
        complex_dtype="complex64",
        doppler_bins=(7, 14),
        doppler_bin_count=7,
        batch_frames=32,
        batches_per_output=64,
        frames_per_output=2048,
        output_stride_frames=32,
        temporal_support_ms=55.35,
        dummy_gil_thread_enabled=False,
        dummy_gil_inner_loops=200_000,
        dummy_gil_iterations=0,
        dummy_gil_iterations_per_second=0.0,
        dummy_gil_switch_interval_s=None,
    )


class ConfigTests(unittest.TestCase):
    def load(self, repetitions: object = None, *, include: bool = False):
        config = copy.deepcopy(BASE_CONFIG)
        if include:
            config["benchmark"]["repetitions"] = repetitions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            return load_benchmark_config(path, implementation_name="test")[0]

    def test_repetitions_default_to_ten(self) -> None:
        self.assertEqual(self.load().benchmark_repetitions, 10)

    def test_explicit_repetitions(self) -> None:
        self.assertEqual(self.load(5, include=True).benchmark_repetitions, 5)

    def test_invalid_repetitions_are_rejected(self) -> None:
        for value in (0, -1, True, 1.5, "10"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                self.load(value, include=True)


class RepetitionTests(unittest.TestCase):
    def test_mode_by_mode_order_and_last_image(self) -> None:
        modes = [
            ExecutionMode("a", True, True),
            ExecutionMode("b", True, True),
        ]
        calls: list[str] = []

        def run(mode: ExecutionMode):
            calls.append(mode.name)
            marker = len(calls)
            return np.array([marker]), make_stats(mode.name, float(marker))

        series = run_repeated_modes(modes=modes, repetitions=3, run_mode=run)

        self.assertEqual(calls, ["a", "a", "a", "b", "b", "b"])
        self.assertEqual([len(item.runs) for item in series], [3, 3])
        np.testing.assert_array_equal(series[0].image, np.array([3]))
        np.testing.assert_array_equal(series[1].image, np.array([6]))

    def test_throughput_summary_uses_sample_standard_deviation(self) -> None:
        series = BenchmarkSeries(
            image=np.zeros(1),
            runs=tuple(make_stats("mode", value) for value in (90.0, 100.0, 110.0)),
        )
        summary = summarize_throughput(series)

        self.assertEqual(summary.mean_input_fps, 100.0)
        self.assertEqual(summary.sample_std_input_fps, 10.0)
        self.assertEqual(summary.coefficient_of_variation_percent, 10.0)
        self.assertEqual(summary.min_input_fps, 90.0)
        self.assertEqual(summary.max_input_fps, 110.0)

    def test_single_run_has_no_sample_spread(self) -> None:
        summary = summarize_throughput(
            BenchmarkSeries(image=np.zeros(1), runs=(make_stats("mode", 100.0),))
        )
        self.assertIsNone(summary.sample_std_input_fps)
        self.assertIsNone(summary.coefficient_of_variation_percent)

    def test_series_rejects_mixed_modes(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkSeries(
                image=np.zeros(1),
                runs=(make_stats("a", 100.0), make_stats("b", 100.0)),
            )


class ReportingAndProfilingTests(unittest.TestCase):
    def test_report_contains_aggregate_and_raw_throughput(self) -> None:
        series = BenchmarkSeries(
            image=np.zeros(1),
            runs=(make_stats("mode", 90.0), make_stats("mode", 110.0)),
        )
        report = format_report([series])

        self.assertIn("Sample SD FPS", report)
        self.assertIn("Raw throughput measurements", report)
        self.assertIn("90.0", report)
        self.assertIn("110.0", report)
        self.assertEqual(report.count("Data types"), 1)

    def test_nsys_generated_config_forces_one_repetition(self) -> None:
        generated = single_mode_config(
            BASE_CONFIG,
            mode=ExecutionMode("test-mode", True, True),
            report_path=Path("report.txt"),
            show_image=False,
        )
        self.assertEqual(generated["benchmark"]["repetitions"], 1)


if __name__ == "__main__":
    unittest.main()
