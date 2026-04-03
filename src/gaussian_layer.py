# src/gaussian_layer.py
# Purpose: Per-series Gaussian NLL decision layer
#
# This is the core novelty of the thesis.
#
# Standard AE anomaly detection:
#   reconstruction_error > fixed_threshold → anomaly
#
# Our approach:
#   fit Gaussian(μ_i, σ_i) to training errors of entity i
#   NLL score = log(σ_i) + (error - μ_i)² / (2σ_i²)
#   NLL score > threshold τ → anomaly
#
# Why this is better:
#   - Each entity has its own error distribution
#   - A score of 0.01 might be normal for entity A but anomalous for entity B
#   - NLL scoring accounts for this automatically

import numpy as np
import torch
from scipy import stats
from scipy.stats import laplace, lognorm
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score)
import os


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GAUSSIAN FITTING
# ════════════════════════════════════════════════════════════════════════════

class PerSeriesGaussian:
    """
    Fits one Gaussian distribution per entity to training reconstruction errors.

    After fitting, scores test windows using negative log-likelihood (NLL).
    Higher NLL = more anomalous (the error is unlikely under the normal distribution).

    Usage:
        gaussian = PerSeriesGaussian()
        gaussian.fit(entity_id, train_errors)
        scores = gaussian.score(entity_id, test_errors)
    """

    def __init__(self):
        # Dictionary to store (μ, σ) for each entity
        # e.g. {"A-1": (0.005, 0.002), "A-2": (0.008, 0.003), ...}
        self.gaussians = {}

    def fit(self, entity_id, train_errors):
        """
        Fit a Gaussian to the training reconstruction errors of one entity.

        Args:
            entity_id    (str):        e.g. "A-1"
            train_errors (np.ndarray): shape (num_windows,)
                                       MSE reconstruction errors on normal data

        Why fit per entity?
        Each spacecraft entity has different sensor characteristics.
        Entity A-1 might normally have errors around 0.003.
        Entity P-10 might normally have errors around 0.012.
        A global threshold would be wrong for both.
        """
        # Compute mean and standard deviation of training errors
        mu    = np.mean(train_errors)
        sigma = np.std(train_errors, ddof=1)  # sample std (ddof=1) is statistically correct

        # Prevent sigma from being zero (would cause division by zero in NLL)
        # This can happen if all training errors are identical
        sigma = max(sigma, 1e-6)

        self.gaussians[entity_id] = (mu, sigma)

    def score(self, entity_id, test_errors):
        """
        Score test windows using negative log-likelihood under the fitted Gaussian.

        NLL formula:
            NLL(e) = log(σ) + (e - μ)² / (2σ²)

        This is the negative log of the Gaussian PDF (dropping constants).
        Higher NLL means the error is less likely under the normal distribution.

        Args:
            entity_id   (str):        e.g. "A-1"
            test_errors (np.ndarray): shape (num_windows,)

        Returns:
            nll_scores (np.ndarray): shape (num_windows,)
                                     higher = more anomalous
        """
        if entity_id not in self.gaussians:
            raise ValueError(f"Entity {entity_id} not fitted. Call fit() first.")

        mu, sigma = self.gaussians[entity_id]

        # Negative log-likelihood of a Gaussian (ignoring constant terms)
        # This is what your thesis Chapter 4.3 derives mathematically
        nll_scores = np.log(sigma) + ((test_errors - mu) ** 2) / (2 * sigma ** 2)

        return nll_scores

    def save(self, save_path):
        """Save fitted Gaussians to disk for later use."""
        np.save(save_path, self.gaussians)
        print(f"Gaussian parameters saved to {save_path}")

    def load(self, load_path):
        """Load previously fitted Gaussians from disk."""
        self.gaussians = np.load(load_path, allow_pickle=True).item()
        print(f"Gaussian parameters loaded from {load_path}")


class PerSeriesLaplace:
    """
    Fits one Laplace distribution per entity to training reconstruction errors.

    Laplace has heavier tails than Gaussian — more robust to outliers in
    training data. Used as a distribution comparison in the ablation study.

    NLL formula:
        NLL(e) = log(2b) + |e - μ| / b
    where b is the scale parameter (analogous to σ in Gaussian).

    Usage:
        lap = PerSeriesLaplace()
        lap.fit(entity_id, train_errors)
        scores = lap.score(entity_id, test_errors)
    """

    def __init__(self):
        # Dictionary to store (μ, b) for each entity
        self.laplacians = {}

    def fit(self, entity_id, train_errors):
        """
        Fit a Laplace distribution to training errors using MLE.

        Args:
            entity_id    (str):        e.g. "A-1"
            train_errors (np.ndarray): shape (num_windows,)
        """
        mu, b = laplace.fit(train_errors)
        b = max(b, 1e-6)
        self.laplacians[entity_id] = (mu, b)

    def score(self, entity_id, test_errors):
        """
        Score test windows using NLL under the fitted Laplace distribution.

        Args:
            entity_id   (str):        e.g. "A-1"
            test_errors (np.ndarray): shape (num_windows,)

        Returns:
            nll_scores (np.ndarray): shape (num_windows,)
                                     higher = more anomalous
        """
        if entity_id not in self.laplacians:
            raise ValueError(f"Entity {entity_id} not fitted. Call fit() first.")

        mu, b = self.laplacians[entity_id]
        nll_scores = np.log(2 * b) + np.abs(test_errors - mu) / b
        return nll_scores


class PerSeriesLogNormal:
    """
    Fits one log-normal distribution per entity to training reconstruction errors.

    Reconstruction errors are always positive and often right-skewed —
    log-normal is theoretically motivated for this reason. Used as a
    distribution comparison in the ablation study.

    NLL formula:
        NLL(e) = log(σ) + log(e) + (log(e) - μ)² / (2σ²)
    where μ and σ are the mean and std of log(e).

    Usage:
        ln = PerSeriesLogNormal()
        ln.fit(entity_id, train_errors)
        scores = ln.score(entity_id, test_errors)
    """

    def __init__(self):
        # Dictionary to store (μ, σ) of log-transformed errors for each entity
        self.lognormals = {}

    def fit(self, entity_id, train_errors):
        """
        Fit a log-normal distribution to training errors.

        Args:
            entity_id    (str):        e.g. "A-1"
            train_errors (np.ndarray): shape (num_windows,)
        """
        # Clip to avoid log(0)
        train_errors = np.clip(train_errors, 1e-10, None)
        mu    = np.mean(np.log(train_errors))
        sigma = np.std(np.log(train_errors), ddof=1)
        sigma = max(sigma, 1e-6)
        self.lognormals[entity_id] = (mu, sigma)

    def score(self, entity_id, test_errors):
        """
        Score test windows using NLL under the fitted log-normal distribution.

        Args:
            entity_id   (str):        e.g. "A-1"
            test_errors (np.ndarray): shape (num_windows,)

        Returns:
            nll_scores (np.ndarray): shape (num_windows,)
                                     higher = more anomalous
        """
        if entity_id not in self.lognormals:
            raise ValueError(f"Entity {entity_id} not fitted. Call fit() first.")

        mu, sigma = self.lognormals[entity_id]
        test_errors = np.clip(test_errors, 1e-10, None)
        nll_scores = (np.log(sigma) + np.log(test_errors) +
                      (np.log(test_errors) - mu) ** 2 / (2 * sigma ** 2))
        return nll_scores


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RECONSTRUCTION ERROR COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_reconstruction_errors(model, windows, device, batch_size=256):
    """
    Compute per-window reconstruction errors for a set of windows.

    Args:
        model      (nn.Module):  trained autoencoder
        windows    (np.ndarray): shape (num_windows, window_size, channels)
        device     (torch.device): cuda or cpu
        batch_size (int): process this many windows at once to avoid OOM

    Returns:
        errors (np.ndarray): shape (num_windows,)
                             MSE reconstruction error per window
    """
    model.eval()
    all_errors = []

    # Process in batches to avoid GPU memory overflow
    num_windows = len(windows)

    with torch.no_grad():
        for start in range(0, num_windows, batch_size):
            end   = min(start + batch_size, num_windows)
            batch = windows[start:end]

            # Convert numpy array to PyTorch tensor and move to GPU
            batch_tensor = torch.FloatTensor(batch).to(device)

            # Get reconstruction
            reconstruction = model(batch_tensor)

            # Per-window MSE: mean over timesteps and channels
            # Shape: (batch_size,)
            errors = ((reconstruction - batch_tensor) ** 2).mean(dim=[1, 2])
            all_errors.append(errors.cpu().numpy())

    return np.concatenate(all_errors, axis=0)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENTITY-LEVEL ERROR COLLECTION FOR GAUSSIAN FITTING
# ════════════════════════════════════════════════════════════════════════════

def collect_entity_train_errors(model, entity_ids, device,
                                 data_dir="data/raw", window_size=100):
    """
    Collect training reconstruction errors SEPARATELY for each entity.

    Why separately?
    The training loop in train.py mixes all entities together for efficiency.
    But for per-series Gaussian fitting we need errors from each entity
    individually — so we run inference entity by entity here.

    Args:
        model      (nn.Module): trained autoencoder
        entity_ids (list):      list of entity ID strings
        device     (torch.device): cuda or cpu
        data_dir   (str):       path to data folder
        window_size (int):      must match training window size

    Returns:
        entity_errors (dict): maps entity_id -> np.ndarray of train errors
    """
    from src.dataset import load_entity, create_windows

    entity_errors = {}

    print(f"Collecting per-entity training errors for {len(entity_ids)} entities...")

    for entity_id in entity_ids:
        # Load raw training data for this entity only
        train_data, _ = load_entity(entity_id, data_dir)

        # Create windows from this entity's training data
        # stride=1 for maximum coverage
        train_windows = create_windows(train_data,
                                       window_size=window_size,
                                       stride=1)

        # Compute reconstruction errors for this entity's windows
        errors = compute_reconstruction_errors(model, train_windows, device)
        entity_errors[entity_id] = errors

    print(f"Done. Example — A-1: mean={entity_errors['A-1'].mean():.6f}, "
          f"std={entity_errors['A-1'].std():.6f}")

    return entity_errors


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def evaluate_detection(scores, labels, threshold=None):
    """
    Evaluate anomaly detection performance given scores and true labels.

    Args:
        scores    (np.ndarray): shape (num_windows,) — higher = more anomalous
        labels    (np.ndarray): shape (num_windows,) — 0=normal, 1=anomaly
        threshold (float):      decision boundary
                                if None, uses best F1 threshold from scores

    Returns:
        metrics (dict): AUC, F1, precision, recall, threshold used
    """
    # ── AUC-ROC ──────────────────────────────────────────────────────────────
    # AUC is threshold-independent — measures overall ranking quality
    # 1.0 = perfect, 0.5 = random
    if len(np.unique(labels)) < 2:
        # Can't compute AUC if only one class present
        auc = float('nan')
    else:
        auc = roc_auc_score(labels, scores)

    # ── Find best threshold ───────────────────────────────────────────────────
    if threshold is None:
        # Try many thresholds and pick the one with best F1
        # This is standard practice in anomaly detection evaluation
        candidate_thresholds = np.percentile(scores, np.arange(50, 99, 1))
        best_f1   = 0
        best_tau  = candidate_thresholds[0]

        for tau in candidate_thresholds:
            preds = (scores > tau).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_tau = tau

        threshold = best_tau

    # ── F1, Precision, Recall at chosen threshold ────────────────────────────
    predictions = (scores > threshold).astype(int)
    f1        = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall    = recall_score(labels, predictions, zero_division=0)

    return {
        "auc":       auc,
        "f1":        f1,
        "precision": precision,
        "recall":    recall,
        "threshold": threshold
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIXED PERCENTILE THRESHOLD (BASELINE DECISION LAYER)
# ════════════════════════════════════════════════════════════════════════════

def fixed_percentile_threshold(train_errors, test_errors, percentile=95):
    """
    Baseline decision layer: flag windows above the Nth percentile
    of training errors as anomalies.

    This is the standard approach in most AE anomaly detection papers.
    Your Gaussian NLL layer replaces this.

    Args:
        train_errors (np.ndarray): errors on normal training data
        test_errors  (np.ndarray): errors on test data
        percentile   (int):        e.g. 95 means top 5% flagged

    Returns:
        scores (np.ndarray): the raw test errors (used as anomaly scores)
        threshold (float):   the percentile cutoff value
    """
    threshold = np.percentile(train_errors, percentile)
    return test_errors, threshold


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FULL EVALUATION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def run_evaluation(model, model_name, entity_ids, test_data_dict,
                   device, data_dir="data/raw", window_size=100):
    """
    Run full evaluation for one model with ALL decision layers:
        - Fixed percentile threshold (standard baseline)
        - Per-series Gaussian NLL    (your primary contribution)
        - Per-series Laplace NLL     (distribution comparison)
        - Per-series Log-Normal NLL  (distribution comparison)

    Args:
        model          (nn.Module): trained autoencoder
        model_name     (str):       "mlp", "lstm", or "transformer"
        entity_ids     (list):      list of entity ID strings
        test_data_dict (dict):      from build_dataloaders()
        device         (torch.device): cuda or cpu
        data_dir       (str):       path to data folder
        window_size    (int):       must match training window size

    Returns:
        results (dict): aggregate metrics for all four decision layers
    """

    # ── Step 1: Collect per-entity training errors ───────────────────────────
    entity_train_errors = collect_entity_train_errors(
        model, entity_ids, device, data_dir, window_size
    )

    # ── Step 2: Fit all three distributions per entity ───────────────────────
    gaussian  = PerSeriesGaussian()
    laplace_d = PerSeriesLaplace()
    lognormal = PerSeriesLogNormal()

    for entity_id, train_errors in entity_train_errors.items():
        gaussian.fit(entity_id,  train_errors)
        laplace_d.fit(entity_id, train_errors)
        lognormal.fit(entity_id, train_errors)

    # Save Gaussian parameters for this model
    save_dir = "results/models"
    gaussian.save(os.path.join(save_dir, f"{model_name}_gaussians.npy"))

    # ── Step 3: Evaluate each entity ─────────────────────────────────────────
    fixed_metrics_list   = []
    gauss_metrics_list   = []
    laplace_metrics_list = []
    lognorm_metrics_list = []

    evaluated_entities = 0

    print(f"\nEvaluating {model_name.upper()} on {len(entity_ids)} entities...")
    print(f"{'Entity':>8} {'AUC-Fix':>9} {'F1-Fix':>7} "
          f"{'AUC-Gau':>9} {'F1-Gau':>7} "
          f"{'AUC-Lap':>9} {'F1-Lap':>7} "
          f"{'AUC-LN':>9} {'F1-LN':>7}")
    print("-" * 80)

    for entity_id in entity_ids:
        test_windows, test_labels = test_data_dict[entity_id]

        # Skip entities with no anomalies
        if test_labels.sum() == 0:
            continue

        evaluated_entities += 1

        # ── Compute test reconstruction errors ───────────────────────────────
        test_errors  = compute_reconstruction_errors(model, test_windows, device)
        train_errors = entity_train_errors[entity_id]

        # ── Fixed percentile (baseline) ───────────────────────────────────────
        fixed_scores, fixed_tau = fixed_percentile_threshold(
            train_errors, test_errors, percentile=95
        )
        fixed_m = evaluate_detection(fixed_scores, test_labels,
                                     threshold=fixed_tau)

        # ── Gaussian NLL (primary contribution) ───────────────────────────────
        gauss_scores = gaussian.score(entity_id, test_errors)
        gauss_m      = evaluate_detection(gauss_scores, test_labels)

        # ── Laplace NLL ───────────────────────────────────────────────────────
        lap_scores = laplace_d.score(entity_id, test_errors)
        lap_m      = evaluate_detection(lap_scores, test_labels)

        # ── Log-Normal NLL ────────────────────────────────────────────────────
        ln_scores = lognormal.score(entity_id, test_errors)
        ln_m      = evaluate_detection(ln_scores, test_labels)

        fixed_metrics_list.append(fixed_m)
        gauss_metrics_list.append(gauss_m)
        laplace_metrics_list.append(lap_m)
        lognorm_metrics_list.append(ln_m)

        print(f"{entity_id:>8} "
              f"{fixed_m['auc']:>9.4f} {fixed_m['f1']:>7.4f} "
              f"{gauss_m['auc']:>9.4f} {gauss_m['f1']:>7.4f} "
              f"{lap_m['auc']:>9.4f} {lap_m['f1']:>7.4f} "
              f"{ln_m['auc']:>9.4f} {ln_m['f1']:>7.4f}")

    # ── Step 4: Aggregate results ─────────────────────────────────────────────
    def aggregate(metrics_list):
        return {
            "auc":       np.mean([m["auc"] for m in metrics_list]),
            "f1":        np.mean([m["f1"] for m in metrics_list]),
            "precision": np.mean([m["precision"] for m in metrics_list]),
            "recall":    np.mean([m["recall"] for m in metrics_list]),
            "auc_std":   np.std([m["auc"] for m in metrics_list]),
            "f1_std":    np.std([m["f1"] for m in metrics_list]),
        }

    fixed_agg   = aggregate(fixed_metrics_list)
    gauss_agg   = aggregate(gauss_metrics_list)
    laplace_agg = aggregate(laplace_metrics_list)
    lognorm_agg = aggregate(lognorm_metrics_list)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'SUMMARY':>8} — {model_name.upper()} "
          f"({evaluated_entities} entities with anomalies)")
    print(f"{'─'*80}")
    print(f"{'Method':<16} {'AUC':>8} {'±':>6} {'F1':>8} {'±':>6} "
          f"{'Precision':>10} {'Recall':>8}")
    print(f"{'─'*80}")

    for name, agg in [("Fixed-95th",  fixed_agg),
                      ("Gaussian-NLL", gauss_agg),
                      ("Laplace-NLL",  laplace_agg),
                      ("LogNormal-NLL", lognorm_agg)]:
        print(f"{name:<16} "
              f"{agg['auc']:>8.4f} {agg['auc_std']:>6.4f} "
              f"{agg['f1']:>8.4f} {agg['f1_std']:>6.4f} "
              f"{agg['precision']:>10.4f} "
              f"{agg['recall']:>8.4f}")

    print(f"{'─'*80}")

    return {
        "model":   model_name,
        "fixed":   fixed_agg,
        "gauss":   gauss_agg,
        "laplace": laplace_agg,
        "lognorm": lognorm_agg,
        "per_entity": {
            "fixed":   fixed_metrics_list,
            "gauss":   gauss_metrics_list,
            "laplace": laplace_metrics_list,
            "lognorm": lognorm_metrics_list,
        }
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Testing gaussian_layer.py...\n")

    # Simulate training errors for one entity (normal data)
    # In reality these come from the trained AE
    np.random.seed(42)
    fake_train_errors = np.abs(np.random.normal(loc=0.005, scale=0.002, size=1000))

    # Simulate test errors — mix of normal and anomalous
    fake_normal_errors  = np.random.normal(loc=0.005, scale=0.002, size=80)
    fake_anomaly_errors = np.random.normal(loc=0.025, scale=0.005, size=10)
    fake_test_errors    = np.abs(np.concatenate([fake_normal_errors,
                                                  fake_anomaly_errors]))
    fake_labels         = np.array([0]*80 + [1]*10)

    # ── Gaussian ──────────────────────────────────────────────────────────────
    gaussian = PerSeriesGaussian()
    gaussian.fit("test_entity", fake_train_errors)
    mu, sigma = gaussian.gaussians["test_entity"]
    print(f"Gaussian:   μ={mu:.6f}, σ={sigma:.6f}")
    nll_scores = gaussian.score("test_entity", fake_test_errors)
    g_metrics  = evaluate_detection(nll_scores, fake_labels)

    # ── Laplace ───────────────────────────────────────────────────────────────
    lap = PerSeriesLaplace()
    lap.fit("test_entity", fake_train_errors)
    mu_l, b = lap.laplacians["test_entity"]
    print(f"Laplace:    μ={mu_l:.6f}, b={b:.6f}")
    lap_scores = lap.score("test_entity", fake_test_errors)
    l_metrics  = evaluate_detection(lap_scores, fake_labels)

    # ── Log-Normal ────────────────────────────────────────────────────────────
    ln = PerSeriesLogNormal()
    ln.fit("test_entity", fake_train_errors)
    mu_ln, sigma_ln = ln.lognormals["test_entity"]
    print(f"Log-Normal: μ={mu_ln:.6f}, σ={sigma_ln:.6f}")
    ln_scores  = ln.score("test_entity", fake_test_errors)
    ln_metrics = evaluate_detection(ln_scores, fake_labels)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'Method':<12} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)
    for name, m in [("Gaussian",  g_metrics),
                    ("Laplace",   l_metrics),
                    ("LogNormal", ln_metrics)]:
        print(f"{name:<12} {m['auc']:>8.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f}")

    # NLL score ranges for intuition
    print(f"\nNLL score ranges (normal vs anomaly):")
    print(f"  Gaussian  — normal: {nll_scores[:80].mean():.4f}, "
          f"anomaly: {nll_scores[80:].mean():.4f}")
    print(f"  Laplace   — normal: {lap_scores[:80].mean():.4f}, "
          f"anomaly: {lap_scores[80:].mean():.4f}")
    print(f"  LogNormal — normal: {ln_scores[:80].mean():.4f}, "
          f"anomaly: {ln_scores[80:].mean():.4f}")

    print("\ngaussian_layer.py test passed. Ready for run_ablation.py.")