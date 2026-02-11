"""Deep learning models: RealMLP and FT-Transformer for tabular data."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RealMLPRegressor(BaseEstimator, RegressorMixin):
    """RealMLP wrapper compatible with sklearn API.

    Uses rtdl-revisiting-models' MLP implementation for tabular data.
    """

    def __init__(
        self,
        d_layers=(256, 256, 128),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        n_epochs=200,
        patience=20,
        random_state=42,
    ):
        self.d_layers = d_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

    def _build_model(self, n_features):
        torch.manual_seed(self.random_state)
        layers = []
        in_dim = n_features
        for d in self.d_layers:
            layers.extend([
                nn.Linear(in_dim, d),
                nn.BatchNorm1d(d),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            in_dim = d
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).to(DEVICE)

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        y_tensor = torch.FloatTensor(np.array(y)).reshape(-1, 1).to(DEVICE)

        self.model_ = self._build_model(X_scaled.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        # Validation data
        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_scaled = self.scaler_.transform(X_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(DEVICE)
            y_val_t = torch.FloatTensor(np.array(y_val)).reshape(-1, 1).to(DEVICE)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.n_epochs):
            self.model_.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Early stopping
            self.model_.eval()
            with torch.no_grad():
                if use_val:
                    val_pred = self.model_(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                else:
                    val_loss = epoch_loss / len(loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"RealMLP: Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model_.eval()
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy().flatten()
        return predictions


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """Feature Tokenizer + Transformer for tabular regression.

    Implements the FT-Transformer architecture.
    """

    def __init__(
        self,
        d_token=64,
        n_heads=4,
        n_blocks=3,
        d_ffn_factor=4.0 / 3.0,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        residual_dropout=0.0,
        lr=1e-4,
        weight_decay=1e-5,
        batch_size=256,
        n_epochs=200,
        patience=20,
        random_state=42,
    ):
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.d_ffn_factor = d_ffn_factor
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.random_state = random_state

    def _build_model(self, n_features):
        torch.manual_seed(self.random_state)
        return _FTTransformerModule(
            n_features=n_features,
            d_token=self.d_token,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        ).to(DEVICE)

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        y_tensor = torch.FloatTensor(np.array(y)).reshape(-1, 1).to(DEVICE)

        self.model_ = self._build_model(X_scaled.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_scaled = self.scaler_.transform(X_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(DEVICE)
            y_val_t = torch.FloatTensor(np.array(y_val)).reshape(-1, 1).to(DEVICE)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.n_epochs):
            self.model_.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.model_.eval()
            with torch.no_grad():
                if use_val:
                    val_pred = self.model_(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                else:
                    val_loss = epoch_loss / len(loader)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"FT-Transformer: Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model_.eval()
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy().flatten()
        return predictions


class _FTTransformerModule(nn.Module):
    """FT-Transformer: each feature gets its own learned embedding, then Transformer."""

    def __init__(
        self,
        n_features,
        d_token=64,
        n_heads=4,
        n_blocks=3,
        d_ffn_factor=4.0 / 3.0,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Feature tokenizer: one linear layer per feature
        self.feature_tokenizer = nn.Linear(n_features, n_features * d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Transformer blocks
        d_ffn = int(d_token * d_ffn_factor)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=attention_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Tokenize: (batch, n_features) -> (batch, n_features, d_token)
        tokens = self.feature_tokenizer(x).view(batch_size, self.n_features, self.d_token)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer
        tokens = self.transformer(tokens)

        # Use CLS token output for prediction
        cls_output = tokens[:, 0, :]
        return self.head(cls_output)


def optimize_realmlp(trial, X, y, cv=5):
    """Optuna objective for RealMLP."""
    from sklearn.model_selection import cross_val_score

    params = {
        "d_layers": trial.suggest_categorical(
            "d_layers",
            [(256, 128), (256, 256, 128), (512, 256, 128), (256, 256, 256, 128)],
        ),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": 100,
        "patience": 15,
        "random_state": 42,
    }
    model = RealMLPRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_ft_transformer(trial, X, y, cv=5):
    """Optuna objective for FT-Transformer."""
    from sklearn.model_selection import cross_val_score

    params = {
        "d_token": trial.suggest_categorical("d_token", [32, 64, 128]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_blocks": trial.suggest_int("n_blocks", 2, 5),
        "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.3),
        "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.3),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_epochs": 100,
        "patience": 15,
        "random_state": 42,
    }
    # d_token must be divisible by n_heads
    if params["d_token"] % params["n_heads"] != 0:
        params["n_heads"] = min(params["n_heads"], params["d_token"])
        while params["d_token"] % params["n_heads"] != 0:
            params["n_heads"] -= 1

    model = FTTransformerRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()
