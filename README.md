# Telemetry Anomaly Detection

> Unsupervised anomaly detection in multivariate spacecraft telemetry using autoencoder-based models with per-series distributional scoring.

---

## What It Does

This project detects anomalies in spacecraft telemetry data without requiring labeled examples. It processes multivariate time series signals from multiple sensor channels simultaneously, reconstructs normal behavior using an autoencoder, and flags anomalies based on per-series distributional scoring — giving channel-level precision on where and when something goes wrong.

---

## Key Highlights

- Fully unsupervised — no labeled anomaly data required during training
- Autoencoder architecture trained on nominal telemetry behavior
- Per-series distributional scoring for fine-grained, channel-level anomaly detection
- Designed for multivariate time series with many concurrent sensor streams
- Evaluated on the real-world NASA SMAP spacecraft telemetry benchmark
- Ablation studies included to assess contribution of each model component

---

## Technologies Used

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

## Dataset

This project uses the **SMAP (Soil Moisture Active Passive)** dataset, a real-world spacecraft telemetry benchmark released by NASA.

- **Source:** NASA / Jet Propulsion Laboratory (JPL)
- **Published by:** Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018
- **Contents:** Telemetry data from the SMAP satellite, including labeled anomaly segments across multiple sensor channels
- **Format:** Multivariate time series — one file per channel, split into `train` and `test` sets
- **Anomaly labels:** Provided for evaluation purposes only; labels are not used during training (unsupervised setting)

### Download
```bash
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip
unzip data.zip
mv data/ data/raw/
```

Or visit the official repository:
> https://github.com/khundman/telemanom

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/telemetry-anomaly-autoencoder.git
cd telemetry-anomaly-autoencoder
```

### 2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Inspect the data
```bash
python inspect_data.py
```

### 4. Train the model
```bash
python train.py
```

### 5. Run anomaly detection and plot results
```bash
python plot_results.py
```

### 6. Run ablation study
```bash
python run_ablation.py
```

---

## Project Structure
