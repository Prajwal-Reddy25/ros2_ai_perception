import pytest
from ros2_ai_perception.metrics import MetricsWindow, summarize


def test_summary_uses_nearest_rank_p95():
    result = summarize(list(range(1, 101)))
    assert result.mean_ms == 50.5
    assert result.median_ms == 50.5
    assert result.p95_ms == 95
    assert result.sample_count == 100


def test_metrics_keeps_timings_distinct_and_bounded():
    metrics = MetricsWindow(max_samples=2)
    metrics.add(5.0, 10.0, 15.0)
    metrics.add(7.0, 14.0, 21.0)
    metrics.add(9.0, 18.0, 27.0)
    snapshot = metrics.snapshot()
    assert snapshot["processed_frame_count"] == 3
    assert snapshot["window_sample_count"] == 2
    assert snapshot["inference"]["mean_ms"] == 8.0
    assert snapshot["processing"]["mean_ms"] == 16.0
    assert snapshot["end_to_end"]["mean_ms"] == 24.0
    assert snapshot["approx_throughput_fps"] == 62.5


def test_metrics_rejects_invalid_values():
    with pytest.raises(ValueError):
        MetricsWindow(0)
    with pytest.raises(ValueError):
        MetricsWindow().add(-1, 2)
