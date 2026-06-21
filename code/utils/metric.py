"""Metrics and timing helpers for DMD2 experiments."""

import time

import numpy as np


def infer_batch_size(batch):
    if batch is None:
        return None
    if isinstance(batch, dict):
        for key in ("image", "images", "real_image", "label"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
    if hasattr(batch, "shape") and len(batch.shape) > 0:
        return int(batch.shape[0])
    return None


def performance_record(step, batch_size, data_time, step_time, extra=None):
    total_time = float(data_time) + float(step_time)
    samples_per_second = 0.0
    if batch_size is not None and total_time > 0:
        samples_per_second = float(batch_size) / total_time

    record = {
        "step": int(step),
        "batch_size": 0 if batch_size is None else int(batch_size),
        "data_time": float(data_time),
        "step_time": float(step_time),
        "total_time": total_time,
        "samples_per_second": samples_per_second,
    }
    if extra:
        record.update(extra)
    return record


class StepTimer:
    # Track data loading and train-step wall time.
    def __init__(self):
        self.reset()

    def reset(self):
        self.data_start = time.time()
        self.step_start = None

    def mark_step_start(self):
        now = time.time()
        data_time = now - self.data_start
        self.step_start = now
        return data_time

    def mark_step_end(self):
        now = time.time()
        step_time = now - self.step_start if self.step_start is not None else 0.0
        self.data_start = now
        return step_time


def compute_feature_stats(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    mu = features.mean(axis=0)
    if features.shape[0] <= 1:
        sigma = np.zeros([features.shape[1], features.shape[1]], dtype=np.float64)
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def save_feature_stats(path, mu, sigma):
    np.savez(path, mu=np.asarray(mu, dtype=np.float64), sigma=np.asarray(sigma, dtype=np.float64))
    return path


def load_feature_stats(path):
    data = np.load(path)
    return np.asarray(data["mu"], dtype=np.float64), np.asarray(data["sigma"], dtype=np.float64)


def load_features(path):
    data = np.load(path)
    if isinstance(data, np.ndarray):
        return data
    if "features" in data:
        return data["features"]
    if "arr_0" in data:
        return data["arr_0"]
    raise KeyError(f"No feature array found in {path}. Expected 'features' or 'arr_0'.")


def matrix_sqrt(mat):
    mat = np.asarray(mat, dtype=np.float64)
    try:
        from scipy import linalg

        result = linalg.sqrtm(mat)
        if np.iscomplexobj(result):
            result = result.real
        return result
    except Exception:
        eigvals, eigvecs = np.linalg.eig(mat)
        eigvals = np.clip(eigvals.real, 0.0, None)
        eigvecs = eigvecs.real
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ np.linalg.pinv(eigvecs)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    covmean = matrix_sqrt(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = matrix_sqrt((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(fid))


def fid_from_features(pred_features, ref_features):
    pred_mu, pred_sigma = compute_feature_stats(pred_features)
    ref_mu, ref_sigma = compute_feature_stats(ref_features)
    return frechet_distance(pred_mu, pred_sigma, ref_mu, ref_sigma)
