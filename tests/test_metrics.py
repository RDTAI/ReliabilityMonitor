import numpy as np
from reliability_monitor.metrics import expected_calibration_error, labeled_metrics, population_stability_index


def test_psi_distinguishes_shift():
    rng = np.random.default_rng(4); reference = rng.normal(size=2000)
    assert population_stability_index(reference, reference.copy()) < 1e-8
    assert population_stability_index(reference, reference + 1.5) > .25


def test_delayed_label_metrics_and_groups():
    report = labeled_metrics({"label": [0, 1, 1, 0], "predicted_class": [0, 1, 0, 0],
                              "confidence": [.9, .8, .8, .7], "covered": [1, 1, 0, 1],
                              "group": ["a", "a", "b", "b"]})
    assert report["accuracy"] == .75
    assert report["coverage"] == .75
    assert report["worst_group_coverage"] == .5
    assert expected_calibration_error([.9, .8], [1, 1]) > 0

