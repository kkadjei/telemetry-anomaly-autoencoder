# src/models.py
# Purpose: all three autoencoder architectures
#   - MLPAutoencoder       (Variant A)
#   - LSTMAutoencoder      (Variant B)
#   - TransformerAutoencoder (Variant C)
#
# All models follow the same interface:
#   forward(x) → reconstruction x̂
#   encode(x)  → latent vector z

import torch
import torch.nn as nn
import math


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MLP AUTOENCODER
# ════════════════════════════════════════════════════════════════════════════

class MLPAutoencoder(nn.Module):
    """
    Vanilla MLP Autoencoder.

    Input:  (batch, window_size, channels) e.g. (64, 100, 25)
    Latent: (batch, latent_dim)            e.g. (64, 16)
    Output: (batch, window_size, channels) e.g. (64, 100, 25)

    How it works:
    1. Flatten the window into a single vector: 100 × 25 = 2500 values
    2. Encoder compresses 2500 → 16 through a series of linear layers
    3. Decoder expands 16 → 2500 through symmetric linear layers
    4. Reshape back to (batch, 100, 25)

    Why flatten?
    MLP layers expect 1D input per sample. Flattening treats the entire
    window as one feature vector — simple but ignores temporal structure.
    This is intentional for Variant A (your weakest deep baseline).
    """

    def __init__(self,
                 window_size=100,
                 n_channels=25,
                 latent_dim=16):
        """
        Args:
            window_size (int): timesteps per window
            n_channels  (int): number of sensor channels
            latent_dim  (int): size of bottleneck (compressed representation)
        """
        # nn.Module.__init__ must always be called first
        # It sets up PyTorch's internal bookkeeping
        super(MLPAutoencoder, self).__init__()

        self.window_size = window_size
        self.n_channels  = n_channels
        self.latent_dim  = latent_dim

        # Input size after flattening
        input_dim = window_size * n_channels  # 100 × 25 = 2500

        # ── Encoder ──────────────────────────────────────────────────────────
        # nn.Sequential chains layers together
        # Data flows through them in order
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # 2500 → 256
            nn.ReLU(),                   # ReLU: max(0, x) — adds nonlinearity
            nn.Linear(256, 64),          # 256 → 64
            nn.ReLU(),
            nn.Linear(64, latent_dim)    # 64 → 16 (bottleneck)
            # No activation on final encoder layer
            # The latent vector can be any real number
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Symmetric to encoder — mirrors the compression in reverse
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),   # 16 → 64
            nn.ReLU(),
            nn.Linear(64, 256),          # 64 → 256
            nn.ReLU(),
            nn.Linear(256, input_dim)    # 256 → 2500
            # No activation on final decoder layer
            # Output can be any real number (data is already normalized 0-1
            # but we don't enforce this — MSE loss handles it)
        )

    def encode(self, x):
        """
        Compress input to latent vector.

        Args:
            x (tensor): shape (batch, window_size, channels)

        Returns:
            z (tensor): shape (batch, latent_dim)
        """
        # Flatten: (batch, 100, 25) → (batch, 2500)
        # -1 means "infer this dimension automatically"
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        # Pass through encoder layers
        z = self.encoder(x_flat)  # shape: (batch, latent_dim)
        return z

    def decode(self, z):
        """
        Reconstruct input from latent vector.

        Args:
            z (tensor): shape (batch, latent_dim)

        Returns:
            x_hat (tensor): shape (batch, window_size, channels)
        """
        # Pass through decoder layers
        out = self.decoder(z)  # shape: (batch, 2500)

        # Reshape back to (batch, window_size, channels)
        x_hat = out.view(-1, self.window_size, self.n_channels)
        return x_hat

    def forward(self, x):
        """
        Full forward pass: encode then decode.

        Args:
            x (tensor): shape (batch, window_size, channels)

        Returns:
            x_hat (tensor): shape (batch, window_size, channels)

        This is called automatically by PyTorch when you do: model(x)
        """
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LSTM AUTOENCODER
# ════════════════════════════════════════════════════════════════════════════

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder.

    Input:  (batch, window_size, channels) e.g. (64, 100, 25)
    Latent: (batch, latent_dim)            e.g. (64, 16)
    Output: (batch, window_size, channels) e.g. (64, 100, 25)

    How it works:
    1. LSTM encoder reads the sequence step by step, building a hidden state
    2. Final hidden state is compressed to latent vector via linear layer
    3. Latent vector is expanded and repeated across time
    4. LSTM decoder reads the repeated vector and reconstructs the sequence

    Why LSTM?
    Unlike MLP which treats all timesteps independently, LSTM has memory —
    each step knows about previous steps. This captures temporal patterns
    like gradual drifts or periodic oscillations in sensor data.
    """

    def __init__(self,
                 window_size=100,
                 n_channels=25,
                 hidden_dim=64,
                 num_layers=2,
                 latent_dim=16):
        """
        Args:
            window_size (int): timesteps per window
            n_channels  (int): number of sensor channels
            hidden_dim  (int): LSTM hidden state size
            num_layers  (int): number of stacked LSTM layers
            latent_dim  (int): size of bottleneck
        """
        super(LSTMAutoencoder, self).__init__()

        self.window_size = window_size
        self.n_channels  = n_channels
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.latent_dim  = latent_dim

        # ── Encoder LSTM ──────────────────────────────────────────────────────
        # batch_first=True means input shape is (batch, seq, features)
        # which matches our (batch, window_size, channels) format
        self.encoder_lstm = nn.LSTM(
            input_size=n_channels,    # 25 channels as input at each timestep
            hidden_size=hidden_dim,   # 64 hidden units
            num_layers=num_layers,    # 2 stacked LSTM layers
            batch_first=True          # input: (batch, seq, features)
        )

        # Compress LSTM hidden state to latent vector
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)  # 64 → 16

        # ── Decoder ───────────────────────────────────────────────────────────
        # Expand latent vector back to LSTM input size
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)  # 16 → 64

        # Decoder LSTM reconstructs the sequence
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Project LSTM output back to channel space
        self.output_fc = nn.Linear(hidden_dim, n_channels)  # 64 → 25

    def encode(self, x):
        """
        Args:
            x (tensor): shape (batch, window_size, channels)

        Returns:
            z (tensor): shape (batch, latent_dim)
        """
        # LSTM returns: output (all hidden states), (h_n, c_n)
        # output shape: (batch, window_size, hidden_dim)
        # h_n shape:    (num_layers, batch, hidden_dim)
        _, (h_n, _) = self.encoder_lstm(x)

        # Take the last layer's hidden state
        # h_n[-1] gives shape: (batch, hidden_dim)
        last_hidden = h_n[-1]

        # Compress to latent
        z = self.encoder_fc(last_hidden)  # shape: (batch, latent_dim)
        return z

    def decode(self, z):
        """
        Args:
            z (tensor): shape (batch, latent_dim)

        Returns:
            x_hat (tensor): shape (batch, window_size, channels)
        """
        # Expand latent to hidden size
        hidden = self.decoder_fc(z)          # shape: (batch, hidden_dim)

        # Repeat hidden vector across all timesteps
        # unsqueeze(1) adds a dimension: (batch, hidden_dim) → (batch, 1, hidden_dim)
        # repeat(1, window_size, 1) copies it window_size times
        hidden_seq = hidden.unsqueeze(1).repeat(1, self.window_size, 1)
        # shape: (batch, window_size, hidden_dim)

        # Pass through decoder LSTM
        lstm_out, _ = self.decoder_lstm(hidden_seq)
        # lstm_out shape: (batch, window_size, hidden_dim)

        # Project to channel space
        x_hat = self.output_fc(lstm_out)     # shape: (batch, window_size, 25)
        return x_hat

    def forward(self, x):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRANSFORMER AUTOENCODER
# ════════════════════════════════════════════════════════════════════════════

class TransformerAutoencoder(nn.Module):
    """
    Transformer Autoencoder.

    Input:  (batch, window_size, channels) e.g. (64, 100, 25)
    Latent: (batch, latent_dim)            e.g. (64, 16)
    Output: (batch, window_size, channels) e.g. (64, 100, 25)

    How it works:
    1. Project channels to d_model dimensions
    2. Add positional encoding (tells model where each timestep is)
    3. Transformer encoder uses self-attention to relate all timesteps
    4. Mean pool across sequence → latent vector
    5. Decoder expands latent back to sequence via linear layers

    Why Transformer?
    Self-attention can relate ANY two timesteps directly, regardless of
    distance. This captures long-range dependencies that LSTM struggles
    with (e.g., a sensor pattern that repeats every 50 steps).
    """

    def __init__(self,
                 window_size=100,
                 n_channels=25,
                 d_model=32,
                 nhead=4,
                 num_layers=2,
                 latent_dim=16):
        """
        Args:
            window_size (int): timesteps per window
            n_channels  (int): number of sensor channels
            d_model     (int): transformer internal dimension
                               must be divisible by nhead
            nhead       (int): number of attention heads
            num_layers  (int): number of transformer encoder layers
            latent_dim  (int): size of bottleneck
        """
        super(TransformerAutoencoder, self).__init__()

        self.window_size = window_size
        self.n_channels  = n_channels
        self.d_model     = d_model
        self.latent_dim  = latent_dim

        # ── Input projection ──────────────────────────────────────────────────
        # Project from n_channels (25) to d_model (32)
        # Transformer needs consistent internal dimensions
        self.input_projection = nn.Linear(n_channels, d_model)  # 25 → 32

        # ── Positional encoding ───────────────────────────────────────────────
        # Transformers have no built-in sense of order
        # We add positional encodings to tell it where each timestep is
        self.positional_encoding = self._build_positional_encoding(
            window_size, d_model
        )

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,       # internal dimension
            nhead=nhead,           # 4 attention heads
            dim_feedforward=128,   # feedforward network size
            batch_first=True,      # input: (batch, seq, features)
            dropout=0.1            # regularization
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers  # 2 stacked layers
        )

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Compress mean-pooled transformer output to latent vector
        self.encoder_fc = nn.Linear(d_model, latent_dim)   # 32 → 16

        # ── Decoder ───────────────────────────────────────────────────────────
        # Simple MLP decoder — expands latent back to full sequence
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model),              # 16 → 32
            nn.ReLU(),
            nn.Linear(d_model, window_size * n_channels) # 32 → 2500
        )

    def _build_positional_encoding(self, window_size, d_model):
        """
        Build sinusoidal positional encoding matrix.

        Why sinusoidal?
        Using sine and cosine functions of different frequencies gives
        each position a unique pattern that the model can learn to use.
        This is the same approach used in the original Transformer paper.

        Returns:
            pe (tensor): shape (1, window_size, d_model)
                         the 1 allows broadcasting across batches
        """
        pe = torch.zeros(window_size, d_model)

        # Position indices: 0, 1, 2, ..., window_size-1
        position = torch.arange(0, window_size).unsqueeze(1).float()

        # Frequency scaling term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (window_size, d_model) → (1, window_size, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer — not a trainable parameter but moves with model
        # to GPU when you call model.to(device)
        return nn.Parameter(pe, requires_grad=False)

    def encode(self, x):
        """
        Args:
            x (tensor): shape (batch, window_size, channels)

        Returns:
            z (tensor): shape (batch, latent_dim)
        """
        # Project channels to d_model
        x_proj = self.input_projection(x)  # (batch, 100, 32)

        # Add positional encoding
        x_proj = x_proj + self.positional_encoding  # (batch, 100, 32)

        # Pass through transformer encoder
        # Self-attention relates all timesteps to each other
        encoded = self.transformer_encoder(x_proj)  # (batch, 100, 32)

        # Mean pool across sequence dimension
        # This collapses (batch, 100, 32) → (batch, 32)
        pooled = encoded.mean(dim=1)

        # Compress to latent
        z = self.encoder_fc(pooled)  # (batch, 16)
        return z

    def decode(self, z):
        """
        Args:
            z (tensor): shape (batch, latent_dim)

        Returns:
            x_hat (tensor): shape (batch, window_size, channels)
        """
        out   = self.decoder(z)                              # (batch, 2500)
        x_hat = out.view(-1, self.window_size, self.n_channels)  # (batch, 100, 25)
        return x_hat

    def forward(self, x):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUICK TEST
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Testing all three models...\n")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create a fake batch to test shapes
    # This mimics what the DataLoader will feed the model
    batch = torch.randn(64, 100, 25).to(device)
    # 64 windows, 100 timesteps, 25 channels

    # ── Test MLP Autoencoder ─────────────────────────────────────────────────
    mlp_ae = MLPAutoencoder(window_size=100, n_channels=25, latent_dim=16).to(device)
    z      = mlp_ae.encode(batch)
    x_hat  = mlp_ae(batch)
    print(f"MLP Autoencoder:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Output shape: {x_hat.shape}")
    params = sum(p.numel() for p in mlp_ae.parameters())
    print(f"  Parameters:   {params:,}\n")

    # ── Test LSTM Autoencoder ────────────────────────────────────────────────
    lstm_ae = LSTMAutoencoder(window_size=100, n_channels=25,
                               hidden_dim=64, num_layers=2,
                               latent_dim=16).to(device)
    z      = lstm_ae.encode(batch)
    x_hat  = lstm_ae(batch)
    print(f"LSTM Autoencoder:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Output shape: {x_hat.shape}")
    params = sum(p.numel() for p in lstm_ae.parameters())
    print(f"  Parameters:   {params:,}\n")

    # ── Test Transformer Autoencoder ─────────────────────────────────────────
    trans_ae = TransformerAutoencoder(window_size=100, n_channels=25,
                                       d_model=32, nhead=4,
                                       num_layers=2, latent_dim=16).to(device)
    z      = trans_ae.encode(batch)
    x_hat  = trans_ae(batch)
    print(f"Transformer Autoencoder:")
    print(f"  Input shape:  {batch.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Output shape: {x_hat.shape}")
    params = sum(p.numel() for p in trans_ae.parameters())
    print(f"  Parameters:   {params:,}\n")

    print("All models working. Ready for training loop.")