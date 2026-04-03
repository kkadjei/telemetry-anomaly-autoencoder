# plot_results.py
# Purpose: generate all thesis figures
# Run with: python plot_results.py
#
# Produces:
#   results/figures/loss_curves.png
#   results/figures/ablation_bars.png
#   results/figures/case_study_*.png  (3 case studies)

import torch
import numpy as np
import os
import json
from src.dataset import get_all_entity_ids, build_dataloaders, load_entity, create_windows
from src.models import MLPAutoencoder, LSTMAutoencoder, TransformerAutoencoder
from src.gaussian_layer import (PerSeriesGaussian, PerSeriesLaplace,
                                  collect_entity_train_errors,
                                  compute_reconstruction_errors,
                                  fixed_percentile_threshold)
from src.attribution import (plot_loss_curves, plot_ablation_bars,
                               plot_case_study)

CONFIG = {
    "window_size": 100,
    "batch_size":  64,
    "latent_dim":  16,
    "data_dir":    "data/raw",
    "models_dir":  "results/models",
    "figures_dir": "results/figures",
    "results_dir": "results",
}

# ── These three entities show interesting anomaly patterns ────────────────────
# A-7: clear point anomaly, NLL works well  (AUC-Gau=0.98)
# D-1: contextual anomaly, NLL works well   (AUC-Gau=0.90)
# E-3: good detection across all methods    (AUC-Gau=0.92)
CASE_STUDY_ENTITIES = ["A-7", "D-1", "E-3"]


def load_model(model_name, config, device):
    """Load a trained model from saved weights."""
    if model_name == "mlp":
        model = MLPAutoencoder(window_size=config["window_size"],
                                n_channels=25, latent_dim=config["latent_dim"])
    elif model_name == "lstm":
        model = LSTMAutoencoder(window_size=config["window_size"],
                                 n_channels=25, hidden_dim=64,
                                 num_layers=2, latent_dim=config["latent_dim"])
    elif model_name == "transformer":
        model = TransformerAutoencoder(window_size=config["window_size"],
                                        n_channels=25, d_model=32, nhead=4,
                                        num_layers=2, latent_dim=config["latent_dim"])

    weights = os.path.join(config["models_dir"], f"{model_name}_best.pt")
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(CONFIG["figures_dir"], exist_ok=True)

    # ── 1. Loss curves ────────────────────────────────────────────────────────
    print("\nPlotting loss curves...")
    plot_loss_curves(CONFIG["models_dir"], CONFIG["figures_dir"])

    # ── 2. Ablation bar chart ─────────────────────────────────────────────────
    print("\nPlotting ablation bars...")
    plot_ablation_bars(
        os.path.join(CONFIG["results_dir"], "ablation_results.json"),
        CONFIG["figures_dir"]
    )

    # ── 3. Case studies ───────────────────────────────────────────────────────
    print("\nGenerating case studies...")

    # Use LSTM + Laplace (your best performing variant)
    model_name = "lstm"
    dist_name  = "Laplace"
    model      = load_model(model_name, CONFIG, device)

    # Load test data
    entity_ids = get_all_entity_ids(spacecraft="SMAP")
    _, _, test_data_dict = build_dataloaders(
        entity_ids,
        data_dir=CONFIG["data_dir"],
        window_size=CONFIG["window_size"],
        batch_size=CONFIG["batch_size"]
    )

    # Collect per-entity train errors and fit distributions
    print("Fitting distributions...")
    entity_train_errors = collect_entity_train_errors(
        model, entity_ids, device,
        CONFIG["data_dir"], CONFIG["window_size"]
    )

    laplace_dist = PerSeriesLaplace()
    gaussian_dist = PerSeriesGaussian()
    for eid, errs in entity_train_errors.items():
        laplace_dist.fit(eid, errs)
        gaussian_dist.fit(eid, errs)

    # Generate one case study per selected entity
    for entity_id in CASE_STUDY_ENTITIES:
        print(f"\nCase study: {entity_id}")
        test_windows, test_labels = test_data_dict[entity_id]

        # Compute test errors
        test_errors  = compute_reconstruction_errors(model, test_windows, device)
        train_errors = entity_train_errors[entity_id]

        # NLL scores using Laplace
        nll_scores = laplace_dist.score(entity_id, test_errors)

        # Fixed scores
        fixed_scores, fixed_threshold = fixed_percentile_threshold(
            train_errors, test_errors, percentile=95
        )

        plot_case_study(
            entity_id=entity_id,
            test_data=test_windows,
            test_labels=test_labels,
            nll_scores=nll_scores,
            fixed_scores=fixed_scores,
            fixed_threshold=fixed_threshold,
            model=model,
            device=device,
            save_dir=CONFIG["figures_dir"],
            model_name=model_name,
            dist_name=dist_name
        )

    print("\nAll figures saved to results/figures/")
    print("\nFigures for your thesis:")
    for f in os.listdir(CONFIG["figures_dir"]):
        print(f"  results/figures/{f}")