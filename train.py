# train.py
# Purpose: train one autoencoder model and save the weights
# Run with: python train.py --model mlp
#           python train.py --model lstm
#           python train.py --model transformer

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
from src.dataset import get_all_entity_ids, build_dataloaders
from src.models import MLPAutoencoder, LSTMAutoencoder, TransformerAutoencoder


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# All hyperparameters in one place
# Change these here rather than hunting through the code
CONFIG = {
    "window_size": 100,      # timesteps per window
    "batch_size":  64,       # windows per training batch
    "latent_dim":  16,       # bottleneck size
    "epochs":      50,       # training epochs
    "lr":          1e-3,     # learning rate for Adam optimizer
    "val_split":   0.1,      # fraction of data for validation
    "data_dir":    "data/raw",
    "save_dir":    "results/models",  # where to save trained weights
}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════════

def get_model(model_name, config):
    """
    Return the right model based on name string.

    Args:
        model_name (str): "mlp", "lstm", or "transformer"
        config     (dict): hyperparameter config

    Returns:
        model (nn.Module): untrained model
    """
    if model_name == "mlp":
        return MLPAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            latent_dim=config["latent_dim"]
        )
    elif model_name == "lstm":
        return LSTMAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            hidden_dim=64,
            num_layers=2,
            latent_dim=config["latent_dim"]
        )
    elif model_name == "transformer":
        return TransformerAutoencoder(
            window_size=config["window_size"],
            n_channels=25,
            d_model=32,
            nhead=4,
            num_layers=2,
            latent_dim=config["latent_dim"]
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose mlp, lstm, or transformer.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one full pass through the training data.

    Args:
        model      (nn.Module):  the autoencoder
        dataloader (DataLoader): training batches
        optimizer  (Optimizer):  Adam optimizer
        criterion  (Loss):       MSE loss function
        device     (torch.device): cuda or cpu

    Returns:
        avg_loss (float): average MSE loss across all batches
    """
    # Set model to training mode
    # This enables dropout and batch normalization if used
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # ── Move data to GPU ─────────────────────────────────────────────────
        # batch shape: (64, 100, 25)
        batch = batch.to(device)

        # ── Forward pass ─────────────────────────────────────────────────────
        # model(batch) calls model.forward(batch) automatically
        reconstruction = model(batch)

        # ── Compute loss ─────────────────────────────────────────────────────
        # MSE loss: average squared difference between input and reconstruction
        # For an autoencoder, input = target (we want to reconstruct the input)
        loss = criterion(reconstruction, batch)

        # ── Backward pass ────────────────────────────────────────────────────
        # Zero gradients from previous batch
        # (PyTorch accumulates gradients by default — we must reset each batch)
        optimizer.zero_grad()

        # Compute gradients via backpropagation
        loss.backward()

        # Update model weights using computed gradients
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set without updating weights.

    Args:
        model      (nn.Module):  trained model
        dataloader (DataLoader): validation batches
        criterion  (Loss):       MSE loss function
        device     (torch.device): cuda or cpu

    Returns:
        avg_loss (float): average validation MSE loss
    """
    # Set model to evaluation mode
    # This disables dropout — we want deterministic output
    model.eval()

    total_loss  = 0.0
    num_batches = 0

    # torch.no_grad() disables gradient computation
    # We don't need gradients during validation — saves memory and time
    with torch.no_grad():
        for batch in dataloader:
            batch          = batch.to(device)
            reconstruction = model(batch)
            loss           = criterion(reconstruction, batch)
            total_loss    += loss.item()
            num_batches   += 1

    avg_loss = total_loss / num_batches
    return avg_loss


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COLLECT TRAINING RECONSTRUCTION ERRORS
# ════════════════════════════════════════════════════════════════════════════

def collect_train_errors(model, dataloader, device):
    """
    After training, collect reconstruction errors for ALL training windows.

    Why do we need this?
    This is the input to your per-series Gaussian layer (Session 5).
    We need the distribution of reconstruction errors on NORMAL data
    so we can fit a Gaussian and detect anomalies at test time.

    Args:
        model      (nn.Module):  trained autoencoder
        dataloader (DataLoader): training data

    Returns:
        errors (np.ndarray): shape (num_windows,)
                             MSE reconstruction error per window
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            batch          = batch.to(device)
            reconstruction = model(batch)

            # Compute per-window MSE
            # (reconstruction - batch)^2 → mean over channels and timesteps
            # Result shape: (batch_size,) — one error value per window
            errors = ((reconstruction - batch) ** 2).mean(dim=[1, 2])
            all_errors.append(errors.cpu().numpy())
            # .cpu() moves tensor from GPU to CPU
            # .numpy() converts PyTorch tensor to numpy array

    # Concatenate list of arrays into one array
    all_errors = np.concatenate(all_errors, axis=0)
    return all_errors


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN TRAINING FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def train(model_name):
    """
    Full training pipeline for one model variant.

    Steps:
    1. Load data
    2. Build model
    3. Train for N epochs
    4. Save model weights
    5. Collect and save training reconstruction errors
    """

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name.upper()} Autoencoder on {device}\n")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    entity_ids = get_all_entity_ids(spacecraft="SMAP")

    train_loader, val_loader, test_data_dict = build_dataloaders(
        entity_ids,
        data_dir=CONFIG["data_dir"],
        window_size=CONFIG["window_size"],
        batch_size=CONFIG["batch_size"],
        val_split=CONFIG["val_split"]
    )

    # ── Build model ──────────────────────────────────────────────────────────
    model = get_model(model_name, CONFIG).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Loss and optimizer ───────────────────────────────────────────────────
    # MSELoss: mean squared error — standard for autoencoders
    criterion = nn.MSELoss()

    # Adam: adaptive learning rate optimizer — standard choice for deep learning
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    # ── Learning rate scheduler ──────────────────────────────────────────────
    # Reduce learning rate by 50% if validation loss stops improving
    # patience=5 means wait 5 epochs before reducing
    # This helps the model converge more precisely in later epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    train_losses    = []
    val_losses      = []

    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Time':>8}")
    print("-" * 45)

    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)

        # Evaluate on validation set
        val_loss = validate(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time

        print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {elapsed:>7.1f}s")

        # ── Save best model ──────────────────────────────────────────────────
        # We save the model whenever validation loss improves
        # This gives us the best version, not just the last version
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG["save_dir"], f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"         ✓ Saved best model (val_loss={best_val_loss:.6f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    # ── Load best model and collect training errors ──────────────────────────
    print("\nCollecting training reconstruction errors...")
    model.load_state_dict(torch.load(save_path))

    train_errors = collect_train_errors(model, train_loader, device)

    # Save errors for use in Gaussian layer (Session 5)
    errors_path = os.path.join(CONFIG["save_dir"], f"{model_name}_train_errors.npy")
    np.save(errors_path, train_errors)
    print(f"Train errors saved: {train_errors.shape}")
    print(f"Error mean: {train_errors.mean():.6f}")
    print(f"Error std:  {train_errors.std():.6f}")

    # Save loss curves for thesis plots
    curves_path = os.path.join(CONFIG["save_dir"], f"{model_name}_loss_curves.npy")
    np.save(curves_path, {
        "train": train_losses,
        "val":   val_losses
    })

    print(f"\nAll files saved to {CONFIG['save_dir']}/")
    print("Ready for Session 5 — Gaussian decision layer.")

    return model, train_errors, test_data_dict


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # argparse lets you pass arguments from the command line
    # e.g. python train.py --model lstm
    parser = argparse.ArgumentParser(description="Train an autoencoder variant")
    parser.add_argument("--model",
                        type=str,
                        default="mlp",
                        choices=["mlp", "lstm", "transformer"],
                        help="Which model to train: mlp, lstm, or transformer")
    args = parser.parse_args()

    train(args.model)