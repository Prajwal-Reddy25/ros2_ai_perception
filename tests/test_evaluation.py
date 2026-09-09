import pytest
from ros2_ai_perception.evaluation import GroundTruth, evaluate_ap50, intersection_over_union
from ros2_ai_perception.types import Detection


def detection(box, score=0.9, label="dog"):
    return Detection(*box, score, 18, label)


def test_intersection_over_union():
    assert intersection_over_union((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert intersection_over_union((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_ap50_perfect_predictions():
    predictions = {"one": (detection((0, 0, 10, 10)),)}
    truths = (GroundTruth("one", "dog", (0, 0, 10, 10)),)
    metrics = evaluate_ap50(predictions, truths)
    assert metrics["map50"] == 1.0
    assert metrics["micro_precision50"] == 1.0
    assert metrics["micro_recall50"] == 1.0


def test_ap50_counts_duplicate_as_false_positive():
    predictions = {
        "one": (
            detection((0, 0, 10, 10), 0.9),
            detection((0, 0, 10, 10), 0.8),
        )
    }
    truths = (GroundTruth("one", "dog", (0, 0, 10, 10)),)
    metrics = evaluate_ap50(predictions, truths)
    assert metrics["micro_precision50"] == 0.5
    assert metrics["micro_recall50"] == 1.0
