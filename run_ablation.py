# run_ablation.py
# Purpose: run the full ablation study across all model variants
#          and produce the final results table for your thesis
#
# Run with: python run_ablation.py
#
# This script:
#   1. Loads all three trained models
#   2. Runs evaluation with all four decision layers
#   3. Prints and saves the complete ablation table

import torch
import numpy as np
import os
import json
from src.dataset import get_all_entity_ids, build_dataloaders
from src.models import MLPAutoencoder, LSTMAutoencoder, TransformerAutoencoder
from src.gaussian_layer import run_evaluation


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "window_size": 100,
    "batch_size":  64,
    "latent_dim":  16,
    "data_dir":    "data/raw",
    "models_dir":  "results/models",
    "results_dir": "results",
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_trained_model(model_name, config, device):
    """
    Load a trained model from saved weights.

    Args:
        model_name (str):         "mlp", "lstm", or "transformer"
        config     (dict):        hyperparameter config
        device     (torch.device): cuda or cpu

    Returns:
        model (nn.Module): model with trained weights loaded
    """
    # Build the correct architecture
    if model_name == "mlp":
        model = MLPAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            latent_dim=config["latent_dim"]
        )
    elif model_name == "lstm":
        model = LSTMAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            hidden_dim=64,
            num_layers=2,
            latent_dim=config["latent_dim"]
        )
    elif model_name == "transformer":
        model = TransformerAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            d_model=32,
            nhead=4,
            num_layers=2,
            latent_dim=config["latent_dim"]
        )

    # Load saved weights
    weights_path = os.path.join(config["models_dir"], f"{model_name}_best.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Move model to GPU
    model = model.to(device)

    # Set to evaluation mode — disables dropout, fixes batch norm
    model.eval()

    print(f"Loaded {model_name.upper()} from {weights_path}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RESULTS TABLE PRINTER
# ════════════════════════════════════════════════════════════════════════════

def print_ablation_table(all_results):
    """
    Print the final thesis ablation table.

    Rows: model × decision layer combinations
    Cols: AUC, F1, Precision, Recall (mean ± std)

    Args:
        all_results (list): list of result dicts from run_evaluation
    """
    print("\n")
    print("═" * 90)
    print("  THESIS ABLATION TABLE — NASA SMAP Anomaly Detection")
    print("  Mean ± Std across all entities with labeled anomalies")
    print("═" * 90)
    print(f"{'Variant':<28} {'AUC':>10} {'F1':>10} "
          f"{'Precision':>12} {'Recall':>10}")
    print("─" * 90)

    for result in all_results:
        model_name = result["model"].upper()

        # Each model has four decision layer variants
        for layer_name, layer_key in [
            ("Fixed-95th",    "fixed"),
            ("Gaussian-NLL",  "gauss"),
            ("Laplace-NLL",   "laplace"),
            ("LogNormal-NLL", "lognorm"),
        ]:
            agg     = result[layer_key]
            variant = f"{model_name} + {layer_name}"

            print(f"{variant:<28} "
                  f"{agg['auc']:>6.4f}±{agg['auc_std']:.3f} "
                  f"{agg['f1']:>6.4f}±{agg['f1_std']:.3f} "
                  f"{agg['precision']:>12.4f} "
                  f"{agg['recall']:>10.4f}")

        print("─" * 90)

    print("═" * 90)
    print("\nKey: NLL = Negative Log-Likelihood decision layer (proposed method)")
    print("     Fixed-95th = 95th percentile threshold (standard baseline)\n")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════

def save_results(all_results, save_dir):
    """
    Save all results to disk for later use in plots and thesis tables.

    Args:
        all_results (list): list of result dicts
        save_dir    (str):  directory to save to
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save as JSON for easy reading
    # We need to convert numpy floats to Python floats for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    clean_results = convert(all_results)
    save_path = os.path.join(save_dir, "ablation_results.json")

    with open(save_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    print(f"Results saved to {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Starting full ablation study...\n")
    print("This will evaluate all 3 models × 4 decision layers")
    print("= 12 variants total\n")

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Load data ────────────────────────────────────────────────────────────
    entity_ids = get_all_entity_ids(spacecraft="SMAP")

    # We only need test_data_dict here — not the training loaders
    _, _, test_data_dict = build_dataloaders(
        entity_ids,
        data_dir=CONFIG["data_dir"],
        window_size=CONFIG["window_size"],
        batch_size=CONFIG["batch_size"]
    )

    # ── Run evaluation for each model ────────────────────────────────────────
    all_results = []

    for model_name in ["mlp", "lstm", "transformer"]:
        print(f"\n{'═'*60}")
        print(f"  Evaluating: {model_name.upper()}")
        print(f"{'═'*60}")

        # Load trained model
        model = load_trained_model(model_name, CONFIG, device)

        # Run full evaluation — all four decision layers
        result = run_evaluation(
            model=model,
            model_name=model_name,
            entity_ids=entity_ids,
            test_data_dict=test_data_dict,
            device=device,
            data_dir=CONFIG["data_dir"],
            window_size=CONFIG["window_size"]
        )

        all_results.append(result)

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── Print final ablation table ────────────────────────────────────────────
    print_ablation_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(all_results, CONFIG["results_dir"])

    print("\nAblation study complete.")
    print("Next step: Session 6 — temporal attribution + plots.")