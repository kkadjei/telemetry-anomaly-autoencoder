# inspect_data.py
# Purpose: understand the structure of NASA SMAP/MSL data
# Run with: python inspect_data.py

import numpy as np
import pandas as pd
import os

# ── 1. Load one training entity ──────────────────────────────────────────────
# os.listdir gives us all filenames in a folder
train_files = os.listdir("data/raw/train")
test_files = os.listdir("data/raw/test")

print("=== Dataset Overview ===")
print(f"Number of train entities: {len(train_files)}")
print(f"Number of test entities:  {len(test_files)}")
print(f"\nFirst 5 train files: {train_files[:5]}")

# ── 2. Load first entity ─────────────────────────────────────────────────────
# np.load loads a .npy file into a numpy array
first_entity = train_files[0].replace(".npy", "")
train_sample = np.load(f"data/raw/train/{train_files[0]}")
test_sample  = np.load(f"data/raw/test/{train_files[0]}")

print(f"\n=== First Entity: {first_entity} ===")
print(f"Train shape: {train_sample.shape}")
print(f"Test shape:  {test_sample.shape}")
# Shape is (timesteps, channels)
# timesteps = number of time points
# channels  = number of sensors

print(f"\nTrain min: {train_sample.min():.4f}")
print(f"Train max: {train_sample.max():.4f}")
print(f"Train mean: {train_sample.mean():.4f}")

# ── 3. Load anomaly labels ───────────────────────────────────────────────────
labels = pd.read_csv("data/raw/labeled_anomalies.csv")
print(f"\n=== Anomaly Labels ===")
print(labels.head(10))
print(f"\nColumns: {list(labels.columns)}")
print(f"Total labeled entities: {len(labels)}")
print(f"Spacecraft types: {labels['spacecraft'].unique()}")

# ── 4. Check one entity's anomaly info ───────────────────────────────────────
sample_label = labels[labels['chan_id'] == first_entity]
if len(sample_label) > 0:
    print(f"\nAnomaly info for {first_entity}:")
    print(sample_label)
else:
    print(f"\n{first_entity} has no labeled anomalies (normal entity)")

print("\nInspection complete. Ready for Session 2.")