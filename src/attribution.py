# src/attribution.py
# Purpose: temporal attribution — which timesteps drove each anomaly
#
# For a flagged window, compute per-timestep reconstruction error
# normalized to attention-style weights. This tells an operator:
# "the anomaly was concentrated in timesteps 60-75 of this window"

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TEMPORAL ATTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

def compute_temporal_attribution(model, window, device):
    """
    Compute per-timestep attribution for one window.

    Method:
        1. Run window through AE → get reconstruction
        2. Compute per-timestep MSE: eₜ = mean((xₜ - x̂ₜ)²) over channels
        3. Normalize: aₜ = eₜ / Σeₜ  ← attention-style weights

    Args:
        model  (nn.Module):  trained autoencoder
        window (np.ndarray): shape (window_size, channels)
        device (torch.device): cuda or cpu

    Returns:
        attribution (np.ndarray): shape (window_size,)
                                  values sum to 1.0
                                  higher = more responsible for anomaly
        reconstruction (np.ndarray): shape (window_size, channels)
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension: (window_size, channels) → (1, window_size, channels)
        x = torch.FloatTensor(window).unsqueeze(0).to(device)

        # Get reconstruction
        x_hat = model(x)

        # Per-timestep MSE averaged over channels
        # Shape: (1, window_size, channels) → (window_size,)
        per_timestep_error = ((x - x_hat) ** 2).mean(dim=2).squeeze(0)
        per_timestep_error = per_timestep_error.cpu().numpy()

    # Normalize to sum to 1 (attention-style weights)
    total = per_timestep_error.sum()
    if total > 0:
        attribution = per_timestep_error / total
    else:
        attribution = np.ones(len(per_timestep_error)) / len(per_timestep_error)

    reconstruction = x_hat.squeeze(0).cpu().numpy()

    return attribution, reconstruction


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CASE STUDY PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_case_study(entity_id, test_data, test_labels,
                    nll_scores, fixed_scores, fixed_threshold,
                    model, device, save_dir,
                    model_name="model", dist_name="Gaussian"):
    """
    Plot a full case study for one entity showing:
        Panel 1: Raw time series (first channel)
        Panel 2: Reconstruction error per window
        Panel 3: NLL anomaly score + threshold
        Panel 4: Fixed percentile score + threshold
        Panel 5: Temporal attribution heatmap for most anomalous window

    Args:
        entity_id       (str):        e.g. "A-7"
        test_data       (np.ndarray): shape (num_windows, window_size, channels)
        test_labels     (np.ndarray): shape (num_windows,)
        nll_scores      (np.ndarray): shape (num_windows,)
        fixed_scores    (np.ndarray): shape (num_windows,)
        fixed_threshold (float):      95th percentile threshold
        model           (nn.Module):  trained autoencoder
        device          (torch.device): cuda or cpu
        save_dir        (str):        where to save the figure
        model_name      (str):        for plot title
        dist_name       (str):        distribution used e.g. "Laplace"
    """
    os.makedirs(save_dir, exist_ok=True)

    num_windows = len(test_data)
    window_indices = np.arange(num_windows)

    # ── Find most anomalous window ────────────────────────────────────────────
    # This is the window we show attribution for
    most_anomalous_idx = np.argmax(nll_scores)

    # ── Compute reconstruction errors per window ──────────────────────────────
    recon_errors = fixed_scores  # fixed_scores = raw MSE errors

    # ── Compute attribution for most anomalous window ─────────────────────────
    anomalous_window = test_data[most_anomalous_idx]
    attribution, reconstruction = compute_temporal_attribution(
        model, anomalous_window, device
    )

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16))
    gs  = gridspec.GridSpec(5, 1, hspace=0.45)

    colors = {
        "normal":    "#2196F3",   # blue
        "anomaly":   "#F44336",   # red
        "threshold": "#FF9800",   # orange
        "nll":       "#9C27B0",   # purple
        "fixed":     "#607D8B",   # grey
        "attr":      "#E91E63",   # pink
    }

    # ── Panel 1: Raw time series ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # Show first channel of the raw test data
    # Flatten windows back to timestep level for visualization
    raw_signal = test_data[:, :, 0].flatten()
    timesteps  = np.arange(len(raw_signal))

    ax1.plot(timesteps, raw_signal, color=colors["normal"],
             linewidth=0.8, alpha=0.8, label="Signal (channel 0)")

    # Shade anomaly windows
    for i, (label, window) in enumerate(zip(test_labels, test_data)):
        if label == 1:
            ax1.axvspan(i * 100, (i + 1) * 100,
                        alpha=0.25, color=colors["anomaly"], label="Anomaly" if i == 0 else "")

    ax1.set_title(f"Entity {entity_id} — Raw Signal (Channel 0)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(0, len(raw_signal))

    # ── Panel 2: Reconstruction error ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    bar_colors = [colors["anomaly"] if l == 1 else colors["normal"]
                  for l in test_labels]
    ax2.bar(window_indices, recon_errors, color=bar_colors,
            alpha=0.7, width=0.8)

    ax2.set_title("Reconstruction Error per Window",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("MSE")

    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["normal"],  alpha=0.7, label="Normal"),
        Patch(facecolor=colors["anomaly"], alpha=0.7, label="Anomaly"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # ── Panel 3: NLL anomaly score ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    ax3.plot(window_indices, nll_scores,
             color=colors["nll"], linewidth=1.2, label=f"{dist_name} NLL Score")

    # Draw best F1 threshold
    # Use 90th percentile of NLL scores as visual threshold
    nll_tau = np.percentile(nll_scores, 90)
    ax3.axhline(y=nll_tau, color=colors["threshold"],
                linestyle="--", linewidth=1.5, label=f"Threshold (90th pct)")

    # Mark anomaly windows
    for i, label in enumerate(test_labels):
        if label == 1:
            ax3.axvspan(i - 0.5, i + 0.5,
                        alpha=0.2, color=colors["anomaly"])

    ax3.set_title(f"NLL Anomaly Score — {model_name.upper()} + {dist_name}",
                  fontsize=12, fontweight="bold")
    ax3.set_ylabel("NLL Score")
    ax3.legend(loc="upper right", fontsize=8)

    # ── Panel 4: Fixed percentile score ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])

    ax4.plot(window_indices, fixed_scores,
             color=colors["fixed"], linewidth=1.2, label="Fixed-95th Score")
    ax4.axhline(y=fixed_threshold, color=colors["threshold"],
                linestyle="--", linewidth=1.5, label=f"95th pct threshold")

    for i, label in enumerate(test_labels):
        if label == 1:
            ax4.axvspan(i - 0.5, i + 0.5,
                        alpha=0.2, color=colors["anomaly"])

    ax4.set_title("Fixed-95th Baseline Score",
                  fontsize=12, fontweight="bold")
    ax4.set_ylabel("MSE Score")
    ax4.legend(loc="upper right", fontsize=8)

    # ── Panel 5: Temporal attribution heatmap ────────────────────────────────
    ax5 = fig.add_subplot(gs[4])

    timestep_indices = np.arange(len(attribution))

    ax5.bar(timestep_indices, attribution,
            color=colors["attr"], alpha=0.8, width=1.0)

    # Overlay the actual signal for the anomalous window
    ax5_twin = ax5.twinx()
    ax5_twin.plot(timestep_indices,
                  anomalous_window[:, 0],
                  color=colors["normal"], linewidth=1.2,
                  alpha=0.7, label="Signal")
    ax5_twin.plot(timestep_indices,
                  reconstruction[:, 0],
                  color=colors["anomaly"], linewidth=1.2,
                  linestyle="--", alpha=0.7, label="Reconstruction")
    ax5_twin.set_ylabel("Signal Value", fontsize=9)
    ax5_twin.legend(loc="upper left", fontsize=8)

    ax5.set_title(
        f"Temporal Attribution — Window {most_anomalous_idx} "
        f"(Most Anomalous, Label={test_labels[most_anomalous_idx]})",
        fontsize=12, fontweight="bold"
    )
    ax5.set_xlabel("Timestep within Window")
    ax5.set_ylabel("Attribution Weight")

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.suptitle(
        f"Case Study: Entity {entity_id} | "
        f"{model_name.upper()} + {dist_name}-NLL",
        fontsize=14, fontweight="bold", y=1.01
    )

    save_path = os.path.join(save_dir, f"case_study_{entity_id}_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOSS CURVE PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(models_dir, save_dir):
    """
    Plot training and validation loss curves for all three models.

    Args:
        models_dir (str): directory containing *_loss_curves.npy files
        save_dir   (str): where to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    model_names = ["mlp", "lstm", "transformer"]
    titles      = ["MLP Autoencoder", "LSTM Autoencoder", "Transformer Autoencoder"]
    colors      = {"train": "#2196F3", "val": "#F44336"}

    for ax, model_name, title in zip(axes, model_names, titles):
        path = os.path.join(models_dir, f"{model_name}_loss_curves.npy")

        # Load loss curves
        # allow_pickle=True needed for dict saved with np.save
        curves = np.load(path, allow_pickle=True).item()
        train_losses = curves["train"]
        val_losses   = curves["val"]
        epochs       = np.arange(1, len(train_losses) + 1)

        ax.plot(epochs, train_losses,
                color=colors["train"], linewidth=1.5,
                label="Train Loss")
        ax.plot(epochs, val_losses,
                color=colors["val"], linewidth=1.5,
                label="Val Loss")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Annotate best val loss
        best_val  = min(val_losses)
        best_epoch = val_losses.index(best_val) + 1
        ax.annotate(f"Best: {best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 3, best_val + 0.001),
                    fontsize=8, color=colors["val"],
                    arrowprops=dict(arrowstyle="->",
                                   color=colors["val"]))

    plt.suptitle("Training and Validation Loss Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ABLATION BAR CHART
# ════════════════════════════════════════════════════════════════════════════

def plot_ablation_bars(results_path, save_dir):
    """
    Plot bar chart comparing all 12 variants on AUC and F1.

    Args:
        results_path (str): path to ablation_results.json
        save_dir     (str): where to save the figure
    """
    import json
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Build data for plotting
    variants   = []
    auc_values = []
    f1_values  = []
    auc_stds   = []
    f1_stds    = []

    layer_map = {
        "fixed":   "Fixed-95th",
        "gauss":   "Gaussian",
        "laplace": "Laplace",
        "lognorm": "LogNormal",
    }

    for result in all_results:
        model = result["model"].upper()
        for key, name in layer_map.items():
            variants.append(f"{model}\n+{name}")
            auc_values.append(result[key]["auc"])
            f1_values.append(result[key]["f1"])
            auc_stds.append(result[key]["auc_std"])
            f1_stds.append(result[key]["f1_std"])

    x      = np.arange(len(variants))
    width  = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # ── AUC plot ──────────────────────────────────────────────────────────────
    bars1 = ax1.bar(x, auc_values, width,
                    yerr=auc_stds, capsize=3,
                    color=["#607D8B", "#9C27B0", "#673AB7", "#3F51B5"] * 3,
                    alpha=0.8)

    ax1.set_ylabel("AUC-ROC", fontsize=11)
    ax1.set_title("AUC-ROC by Model Variant",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.axhline(y=0.5, color="red", linestyle="--",
                alpha=0.5, label="Random baseline (0.5)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars1, auc_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ── F1 plot ───────────────────────────────────────────────────────────────
    bars2 = ax2.bar(x, f1_values, width,
                    yerr=f1_stds, capsize=3,
                    color=["#607D8B", "#9C27B0", "#673AB7", "#3F51B5"] * 3,
                    alpha=0.8)

    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_title("F1 Score by Model Variant",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, fontsize=8)
    ax2.set_ylim(0, 0.7)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars2, f1_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Ablation Study — All 12 Variants",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, "ablation_bars.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")