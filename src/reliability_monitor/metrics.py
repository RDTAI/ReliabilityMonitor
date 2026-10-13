"""Transparent monitoring metrics with no training-framework dependency."""
from __future__ import annotations
import numpy as np


def _array(x, ndim=1):
    out = np.asarray(x, dtype=float)
    if ndim == 2 and out.ndim == 1:
        out = out[:, None]
    return out


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI using reference quantile bins; larger values indicate drift."""
    ref, cur = _array(reference), _array(current)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rp = np.histogram(ref, edges)[0] / max(len(ref), 1)
    cp = np.histogram(cur, edges)[0] / max(len(cur), 1)
    rp, cp = np.clip(rp, 1e-6, None), np.clip(cp, 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def categorical_total_variation(reference, current) -> float:
    labels = np.union1d(reference, current)
    rp = np.array([(np.asarray(reference) == x).mean() for x in labels])
    cp = np.array([(np.asarray(current) == x).mean() for x in labels])
    return float(0.5 * np.abs(rp - cp).sum())


def embedding_mean_shift(reference, current) -> float:
    """L2 distance between means, normalized by reference feature scale."""
    ref, cur = _array(reference, 2), _array(current, 2)
    scale = np.std(ref, axis=0) + 1e-8
    return float(np.linalg.norm((cur.mean(0) - ref.mean(0)) / scale) / np.sqrt(ref.shape[1]))


def expected_calibration_error(confidence, correct, bins: int = 10) -> float:
    confidence, correct = _array(confidence), _array(correct)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def confidence_error_curve(confidence, correct, bins: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            rows.append({"lower": float(lo), "upper": float(hi), "n": int(mask.sum()),
                         "mean_confidence": float(confidence[mask].mean()),
                         "error_rate": float(1 - correct[mask].mean())})
    return rows


def unlabeled_metrics(reference: dict, current: dict) -> dict:
    ref_x, cur_x = _array(reference["features"], 2), _array(current["features"], 2)
    feature_psi = [population_stability_index(ref_x[:, j], cur_x[:, j]) for j in range(ref_x.shape[1])]
    return {
        "n_unlabeled": int(len(cur_x)),
        "feature_psi_max": float(max(feature_psi, default=0)),
        "feature_psi_mean": float(np.mean(feature_psi)),
        "class_ratio_shift": categorical_total_variation(reference["predicted_class"], current["predicted_class"]),
        "confidence_psi": population_stability_index(reference["confidence"], current["confidence"]),
        "embedding_shift": embedding_mean_shift(reference["embeddings"], current["embeddings"]),
        "ood_rate": float(np.mean(current["ood"])),
        "mean_set_size": float(np.mean(current["set_size"])),
        "empty_set_rate": float(np.mean(np.asarray(current["set_size"]) == 0)),
    }


def labeled_metrics(current: dict, target_coverage: float = 0.9) -> dict:
    y = np.asarray(current["label"], int)
    pred = np.asarray(current["predicted_class"], int)
    conf = _array(current["confidence"])
    covered = np.asarray(current["covered"], bool)
    correct = pred == y
    result = {
        "n_labeled": int(len(y)),
        "accuracy": float(correct.mean()),
        "coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - target_coverage),
        "ece": expected_calibration_error(conf, correct),
        "brier_binary_correctness": float(np.mean((conf - correct) ** 2)),
        "confidence_error_curve": confidence_error_curve(conf, correct),
    }
    groups = np.asarray(current.get("group", ["all"] * len(y)), str)
    group_cov = {str(g): float(covered[groups == g].mean()) for g in np.unique(groups)}
    group_acc = {str(g): float(correct[groups == g].mean()) for g in np.unique(groups)}
    result.update({"subgroup_coverage": group_cov, "subgroup_accuracy": group_acc,
                   "worst_group_coverage": min(group_cov.values()) if group_cov else None})
    return result

