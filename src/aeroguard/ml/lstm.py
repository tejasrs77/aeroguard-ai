"""Small, reproducible PyTorch LSTM for RUL sequence regression."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aeroguard.config import DEFAULT_RUL_CAP, RANDOM_STATE
from aeroguard.ml.metrics import regression_metrics
from aeroguard.ml.sequences import SequenceBatch


@dataclass(frozen=True)
class LSTMConfig:
    input_size: int
    hidden_size: int = 48
    num_layers: int = 2
    dropout: float = 0.15
    dense_size: int = 24

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RULLSTM(nn.Module):
    """Encode a sensor window and return one non-negative RUL estimate."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.dense_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_size, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.head(encoded[:, -1, :]).squeeze(1)


@dataclass
class LSTMTrainingResult:
    model: RULLSTM
    history: pd.DataFrame
    best_epoch: int
    validation_metrics: dict[str, float]
    device: str


def set_reproducible_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> str:
    """Use an accelerator when available and otherwise use the CPU."""

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_lstm(
    model: RULLSTM,
    values: np.ndarray,
    *,
    batch_size: int = 512,
    device: str = "cpu",
    rul_cap: int = DEFAULT_RUL_CAP,
) -> np.ndarray:
    """Predict in batches and enforce the target's valid range."""

    model = model.to(device)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values.astype("float32"))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            outputs.append(model(batch.to(device)).cpu().numpy())
    return np.clip(np.concatenate(outputs), 0, rul_cap)


def _train_epoch(
    model: RULLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for values, targets in loader:
        values = values.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(values)
        loss = loss_function(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(values)
        total_rows += len(values)
    return total_loss / total_rows


def train_lstm(
    training: SequenceBatch,
    validation: SequenceBatch,
    config: LSTMConfig,
    *,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = RANDOM_STATE,
    device: str | None = None,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> LSTMTrainingResult:
    """Train with early stopping based only on unseen validation engines."""

    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    set_reproducible_seed(seed)
    selected_device = device or choose_device()
    model = RULLSTM(config).to(selected_device)
    dataset = TensorDataset(
        torch.from_numpy(training.values),
        torch.from_numpy(training.targets),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.HuberLoss(delta=10.0)

    best_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        training_loss = _train_epoch(
            model,
            loader,
            optimizer,
            loss_function,
            selected_device,
        )
        validation_predictions = predict_lstm(
            model,
            validation.values,
            device=selected_device,
            rul_cap=rul_cap,
        )
        metrics = regression_metrics(validation.targets, validation_predictions)
        history.append(
            {
                "epoch": epoch,
                "training_huber_loss": training_loss,
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
            }
        )
        print(
            f"    epoch {epoch:02d}: training loss={training_loss:.3f}, "
            f"validation MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}"
        )
        if metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"    early stopping: best epoch was {best_epoch}")
                break

    model.load_state_dict(best_state)
    best_predictions = predict_lstm(
        model,
        validation.values,
        device=selected_device,
        rul_cap=rul_cap,
    )
    return LSTMTrainingResult(
        model=model.cpu(),
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        validation_metrics=regression_metrics(validation.targets, best_predictions),
        device=selected_device,
    )


def fit_lstm_fixed_epochs(
    training: SequenceBatch,
    config: LSTMConfig,
    *,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = RANDOM_STATE,
    device: str | None = None,
) -> RULLSTM:
    """Refit on all training engines for the validation-selected epoch count."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    set_reproducible_seed(seed)
    selected_device = device or choose_device()
    model = RULLSTM(config).to(selected_device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training.values),
            torch.from_numpy(training.targets),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = nn.HuberLoss(delta=10.0)
    for _ in range(epochs):
        _train_epoch(model, loader, optimizer, loss_function, selected_device)
    return model.cpu()


def load_lstm_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
) -> RULLSTM:
    """Reconstruct an inference model from the portable Day 3 checkpoint."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = LSTMConfig(**checkpoint["model_config"])
    model = RULLSTM(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model
