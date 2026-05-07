"""Training and inference utilities for the pricing neural network."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_option_pricing.config import TrainingConfig
from nn_option_pricing.dataset import FEATURE_COLUMNS, TARGET_COLUMN
from nn_option_pricing.model import PricingMLP


@dataclass
class TrainingArtifacts:
    """Objects produced by a training run.

    Attributes
    ----------
    model
        Trained pricing network loaded with the best validation state.
    scaler
        Fitted input feature scaler.
    target_scaler
        Fitted target scaler used to train on normalized prices.
    history
        Per-epoch training and validation losses.
    x_test
        Scaled test features, ready for model inference.
    y_test
        Original-scale Black-Scholes test prices.
    """

    model: PricingMLP
    scaler: StandardScaler
    target_scaler: StandardScaler
    history: dict[str, list[float]]
    x_test: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible NumPy and PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    """Build a PyTorch DataLoader from NumPy feature and target arrays."""
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(
    df, config: TrainingConfig, device: str | None = None
) -> TrainingArtifacts:
    """Train a feed-forward neural network on Black-Scholes prices.

    Parameters
    ----------
    df
        DataFrame containing feature columns and the analytical call price
        target.
    config
        Training hyperparameters and split proportions.
    device
        Optional PyTorch device name. If omitted, CUDA is used when available
        and CPU otherwise.

    Returns
    -------
    TrainingArtifacts
        Trained model, fitted scalers, loss history, and held-out test data.
    """
    set_seed(config.seed)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    # First reserve the test set. The remaining data are split again into
    # training and validation sets, so the test set stays completely untouched
    # until final evaluation.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
    )

    validation_fraction = config.validation_size / (1.0 - config.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=config.seed,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Option prices can span a wider range than the normalized inputs. Scaling
    # the target makes the MSE optimization better conditioned; predictions are
    # mapped back to price units during inference.
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    train_loader = _make_loader(
        x_train_scaled, y_train_scaled, config.batch_size, shuffle=True
    )
    val_loader = _make_loader(
        x_val_scaled, y_val_scaled, config.batch_size, shuffle=False
    )

    model = PricingMLP(input_dim=x.shape[1], hidden_layers=config.hidden_layers).to(
        torch_device
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _epoch in range(config.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(torch_device)
            yb = yb.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(xb)
            loss = criterion(predictions, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(torch_device)
                yb = yb.to(torch_device)
                predictions = model(xb)
                loss = criterion(predictions, yb)
                val_loss_sum += loss.item() * len(xb)
                val_count += len(xb)

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Store a CPU copy of the best state to make early stopping
            # independent of the current device and future optimizer updates.
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        target_scaler=target_scaler,
        history=history,
        x_test=x_test_scaled,
        y_test=y_test,
    )


def predict(
    model: PricingMLP,
    x_scaled: np.ndarray,
    target_scaler: StandardScaler | None = None,
    batch_size: int = 8192,
    device: str | None = None,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Predict option prices with a trained model.

    Parameters
    ----------
    model
        Trained pricing network.
    x_scaled
        Scaled input features, typically produced by the fitted input scaler.
    target_scaler
        Optional fitted target scaler. When provided, predictions are converted
        back to original price units.
    batch_size
        Number of rows processed per inference batch.
    device
        Optional PyTorch device name.
    clip_nonnegative
        Whether to enforce the financial constraint that call prices cannot be
        negative.

    Returns
    -------
    np.ndarray
        Predicted option prices.
    """
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(torch_device)
    model.eval()

    predictions: list[np.ndarray] = []
    dataset = TensorDataset(torch.as_tensor(x_scaled, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(torch_device)
            predictions.append(model(xb).cpu().numpy())

    y_pred = np.concatenate(predictions)
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    if clip_nonnegative:
        # The neural network output is unconstrained. Clipping prevents small
        # negative values after inverse scaling, which would be invalid prices.
        y_pred = np.maximum(y_pred, 0.0)
    return y_pred
