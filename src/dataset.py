# src/dataset.py
# Purpose: load NASA SMAP/MSL data, create sliding windows,
#          and build PyTorch DataLoaders for training and testing
#
# This file is the foundation everything else builds on.
# Read it fully before moving to Session 3.

import numpy as np
import pandas as pd
import ast
import os
import torch
from torch.utils.data import Dataset, DataLoader

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD RAW DATA
# ════════════════════════════════════════════════════════════════════════════

def load_entity(entity_id, data_dir="data/raw"):
    """
    Load train and test numpy arrays for one entity.

    Args:
        entity_id (str): e.g. "A-1", "P-1", "M-1"
        data_dir  (str): path to folder containing train/ and test/

    Returns:
        train_data (np.ndarray): shape (timesteps, channels)
        test_data  (np.ndarray): shape (timesteps, channels)
    """
    # np.load reads a .npy file from disk into a numpy array
    train_path = os.path.join(data_dir, "train", f"{entity_id}.npy")
    test_path  = os.path.join(data_dir, "test",  f"{entity_id}.npy")

    train_data = np.load(train_path)  # shape: (timesteps, 25)
    test_data  = np.load(test_path)   # shape: (timesteps, 25)

    return train_data, test_data


def load_labels(data_dir="data/raw"):
    """
    Load anomaly labels for all entities.

    Returns:
        labels (dict): maps entity_id -> list of [start, end] anomaly intervals
                       e.g. {"A-1": [[4690, 4774]], "P-1": [[2149, 2349], ...]}
    """
    labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
    df = pd.read_csv(labels_path)

    labels = {}
    for _, row in df.iterrows():
        entity_id = row['chan_id']

        # anomaly_sequences is stored as a string like "[[2149, 2349], [4536, 4844]]"
        # ast.literal_eval safely converts that string into a real Python list
        sequences = ast.literal_eval(row['anomaly_sequences'])
        labels[entity_id] = sequences

    return labels


def get_all_entity_ids(data_dir="data/raw", spacecraft=None):
    """
    Get list of all entity IDs from the train folder.

    Args:
        data_dir   (str):  path to data folder
        spacecraft (str):  filter by spacecraft type
                           "SMAP" → only SMAP entities (25 channels)
                           "MSL"  → only MSL entities (55 channels)
                           None   → all entities

    Returns:
        entity_ids (list): e.g. ["A-1", "A-2", ..., "P-1", ...]
    """
    train_files = os.listdir(os.path.join(data_dir, "train"))
    entity_ids = [f.replace(".npy", "") for f in train_files if f.endswith(".npy")]
    entity_ids.sort()

    # Filter by spacecraft type if requested
    if spacecraft is not None:
        labels_path = os.path.join(data_dir, "labeled_anomalies.csv")
        df = pd.read_csv(labels_path)

        # Get entity IDs belonging to the requested spacecraft
        filtered = df[df['spacecraft'] == spacecraft]['chan_id'].tolist()
        entity_ids = [e for e in entity_ids if e in filtered]

    return entity_ids


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SLIDING WINDOW
# ════════════════════════════════════════════════════════════════════════════

def create_windows(data, window_size=100, stride=1):
    """
    Convert a time series into overlapping windows.

    Why windowing?
    Autoencoders expect fixed-size inputs. A time series of 2880 timesteps
    needs to be cut into chunks. We use a sliding window that moves one
    step at a time (stride=1) to maximize training data.

    Args:
        data        (np.ndarray): shape (timesteps, channels)
        window_size (int):        number of timesteps per window (default 100)
        stride      (int):        how many steps to move between windows
                                  stride=1 for training (maximum overlap)
                                  stride=window_size for testing (no overlap)

    Returns:
        windows (np.ndarray): shape (num_windows, window_size, channels)

    Example:
        data has 2880 timesteps, window_size=100, stride=1
        → produces 2781 windows of shape (100, 25)
    """
    timesteps = data.shape[0]
    windows = []

    # Start at 0, move forward by stride each time
    # Stop when we can no longer fit a full window
    for start in range(0, timesteps - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]   # shape: (window_size, channels)
        windows.append(window)

    # Stack list of arrays into one array
    windows = np.array(windows)    # shape: (num_windows, window_size, channels)

    return windows.astype(np.float32)
    # float32 is important — PyTorch models expect float32, not float64


def create_test_labels(test_data, anomaly_intervals, window_size=100, stride=100):
    """
    Create binary labels (0=normal, 1=anomaly) for each test window.

    A window is labeled as anomaly if ANY timestep inside it falls
    within a known anomaly interval.

    Args:
        test_data         (np.ndarray): shape (timesteps, channels)
        anomaly_intervals (list):       e.g. [[4690, 4774], [5000, 5100]]
        window_size       (int):        must match what you used in create_windows
        stride            (int):        must match what you used in create_windows

    Returns:
        window_labels (np.ndarray): shape (num_windows,) with values 0 or 1
    """
    timesteps = test_data.shape[0]
    window_labels = []

    # Create a timestep-level label array first
    # Start with all zeros (normal)
    timestep_labels = np.zeros(timesteps, dtype=int)

    # Mark anomaly timesteps as 1
    for start, end in anomaly_intervals:
        # end is inclusive in the dataset, so we use end+1 for Python slicing
        timestep_labels[start:end+1] = 1

    # Now assign a label to each window
    for start in range(0, timesteps - window_size + 1, stride):
        end = start + window_size
        window_slice = timestep_labels[start:end]

        # If ANY timestep in this window is anomalous, label the window as 1
        label = 1 if window_slice.sum() > 0 else 0
        window_labels.append(label)

    return np.array(window_labels, dtype=int)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PYTORCH DATASET CLASS
# ════════════════════════════════════════════════════════════════════════════

class SMAPDataset(Dataset):
    """
    PyTorch Dataset for NASA SMAP/MSL windows.

    What is a PyTorch Dataset?
    It is a class that tells PyTorch how to access your data.
    You must implement two methods:
      __len__  → how many samples are in the dataset
      __getitem__ → how to get sample number i

    PyTorch's DataLoader will call these automatically during training.
    """

    def __init__(self, windows):
        """
        Args:
            windows (np.ndarray): shape (num_windows, window_size, channels)
        """
        # Convert numpy array to PyTorch tensor
        # torch.FloatTensor creates a float32 tensor, which is what
        # neural networks expect
        self.windows = torch.FloatTensor(windows)

    def __len__(self):
        # Returns total number of windows
        # DataLoader uses this to know when one epoch is complete
        return len(self.windows)

    def __getitem__(self, idx):
        # Returns one window at index idx
        # DataLoader calls this repeatedly to build batches
        return self.windows[idx]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD DATALOADERS
# ════════════════════════════════════════════════════════════════════════════

def build_dataloaders(entity_ids,
                      data_dir="data/raw",
                      window_size=100,
                      batch_size=64,
                      val_split=0.1):
    """
    Build train, validation, and test DataLoaders for all entities combined.

    Why combine all entities for training?
    We want ONE shared AE that learns general normal behavior across
    all 82 spacecraft entities. The per-series Gaussian layer handles
    entity-specific adaptation at inference time.

    Args:
        entity_ids  (list):  list of entity ID strings
        data_dir    (str):   path to data folder
        window_size (int):   timesteps per window
        batch_size  (int):   windows per batch during training
        val_split   (float): fraction of train windows used for validation

    Returns:
        train_loader  (DataLoader): batched normal windows for training
        val_loader    (DataLoader): batched normal windows for validation
        test_data_dict (dict):      maps entity_id -> (test_windows, labels)
    """

    all_train_windows = []
    test_data_dict    = {}
    labels_dict       = load_labels(data_dir)

    print(f"Loading {len(entity_ids)} entities...")

    for entity_id in entity_ids:
        # ── Load raw data ────────────────────────────────────────────────────
        train_data, test_data = load_entity(entity_id, data_dir)

        # ── Create training windows (stride=1 for maximum overlap) ───────────
        train_windows = create_windows(train_data,
                                       window_size=window_size,
                                       stride=1)
        all_train_windows.append(train_windows)

        # ── Create test windows (stride=window_size, no overlap) ─────────────
        # For evaluation we use non-overlapping windows so each timestep
        # is evaluated exactly once
        test_windows = create_windows(test_data,
                                      window_size=window_size,
                                      stride=window_size)

        # ── Create test labels ───────────────────────────────────────────────
        if entity_id in labels_dict:
            anomaly_intervals = labels_dict[entity_id]
        else:
            anomaly_intervals = []  # no anomalies for this entity

        test_labels = create_test_labels(test_data,
                                         anomaly_intervals,
                                         window_size=window_size,
                                         stride=window_size)

        test_data_dict[entity_id] = (test_windows, test_labels)

    # ── Combine all train windows ────────────────────────────────────────────
    # np.concatenate joins arrays along axis 0 (the windows axis)
    all_train_windows = np.concatenate(all_train_windows, axis=0)
    print(f"Total training windows: {len(all_train_windows)}")

    # ── Train / validation split ─────────────────────────────────────────────
    # We split by index, not randomly, to respect temporal order
    n_val   = int(len(all_train_windows) * val_split)
    n_train = len(all_train_windows) - n_val

    train_windows = all_train_windows[:n_train]
    val_windows   = all_train_windows[n_train:]

    print(f"Train windows: {len(train_windows)}")
    print(f"Val windows:   {len(val_windows)}")

    # ── Build PyTorch Datasets ───────────────────────────────────────────────
    train_dataset = SMAPDataset(train_windows)
    val_dataset   = SMAPDataset(val_windows)

    # ── Build DataLoaders ────────────────────────────────────────────────────
    # DataLoader wraps a Dataset and handles:
    #   - batching (grouping windows into batches of batch_size)
    #   - shuffling (randomizing order each epoch)
    #   - parallel loading (num_workers)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,      # shuffle each epoch
                              num_workers=0)     # 0 = load on main thread
                                                 # (safest on Windows)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,     # no shuffle for validation
                            num_workers=0)

    return train_loader, val_loader, test_data_dict


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # It will not run when other files import from this file

    print("Testing dataset.py...\n")

    # Get all entity IDs
    entity_ids = get_all_entity_ids(spacecraft="SMAP")
    print(f"Found {len(entity_ids)} entities: {entity_ids[:5]}...\n")

    # Build dataloaders
    train_loader, val_loader, test_data_dict = build_dataloaders(
        entity_ids,
        window_size=100,
        batch_size=64
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    print(f"\nOne training batch shape: {batch.shape}")
    # Expected: torch.Size([64, 100, 25])
    # 64 windows, each 100 timesteps, each with 25 channels

    # Inspect test data for one entity
    entity = entity_ids[0]
    test_windows, test_labels = test_data_dict[entity]
    print(f"\nTest entity: {entity}")
    print(f"Test windows shape: {test_windows.shape}")
    print(f"Test labels shape:  {test_labels.shape}")
    print(f"Anomaly rate: {test_labels.mean()*100:.1f}%")

    print("\nSession 2 complete. Ready for Session 3.")