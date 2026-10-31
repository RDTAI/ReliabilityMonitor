import numpy as np
from reliability_monitor.metrics import (expected_calibration_error, labeled_metrics,
                                         population_stability_index, wilson_interval)


def test_psi_distinguishes_shift():
    rng = np.random.default_rng(4); reference = rng.normal(size=2000)
    assert population_stability_index(reference, reference.copy()) < 1e-8
    assert population_stability_index(reference, reference + 1.5) > .25


def test_delayed_label_metrics_and_groups():
    report = labeled_metrics({"label": [0, 1, 1, 0], "predicted_class": [0, 1, 0, 0],
                              "confidence": [.9, .8, .8, .7], "covered": [1, 1, 0, 1],
                              "group": ["a", "a", "b", "b"]}, min_group_size=1)
    assert report["accuracy"] == .75
    assert report["coverage"] == .75
    assert report["worst_group_coverage"] == .5
    assert expected_calibration_error([.9, .8], [1, 1]) > 0


def test_wilson_interval_and_small_group_guard():
    low, high = wilson_interval(8, 10)
    assert 0 < low < .8 < high < 1
    report = labeled_metrics({"label": [0, 0], "predicted_class": [0, 0],
                              "confidence": [.9, .9], "covered": [1, 1],
                              "group": ["tiny", "tiny"]}, min_group_size=20)
    assert report["worst_group_coverage"] is None
    assert report["subgroup_reliability"]["tiny"]["status"] == "insufficient_data"

