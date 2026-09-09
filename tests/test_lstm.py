from __future__ import annotations

import numpy as np
import torch

from aeroguard.ml.lstm import (
    LSTMConfig,
    RULLSTM,
    load_lstm_checkpoint,
    predict_lstm,
    train_lstm,
)
from aeroguard.ml.sequences import SequenceBatch


def _tiny_batch(samples: int, seed: int) -> SequenceBatch:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(samples, 4, 3)).astype("float32")
    targets = np.maximum(0, 20 + values[:, -1, :].sum(axis=1)).astype("float32")
    return SequenceBatch(
        values=values,
        targets=targets,
        engine_ids=np.arange(1, samples + 1),
        cycles=np.full(samples, 4),
    )


def test_lstm_forward_returns_one_non_negative_value_per_sequence() -> None:
    model = RULLSTM(LSTMConfig(input_size=3, hidden_size=8, num_layers=1))

    output = model(torch.zeros((5, 4, 3)))

    assert output.shape == (5,)
    assert torch.all(output >= 0)


def test_lstm_training_and_prediction_smoke() -> None:
    training = _tiny_batch(20, 1)
    validation = _tiny_batch(8, 2)
    config = LSTMConfig(input_size=3, hidden_size=8, num_layers=1, dense_size=4)

    result = train_lstm(
        training,
        validation,
        config,
        max_epochs=2,
        patience=2,
        batch_size=8,
        device="cpu",
    )
    predictions = predict_lstm(result.model, validation.values)

    assert 1 <= result.best_epoch <= 2
    assert len(result.history) == 2
    assert predictions.shape == (8,)
    assert np.all((predictions >= 0) & (predictions <= 125))


def test_lstm_checkpoint_round_trip(tmp_path) -> None:
    config = LSTMConfig(input_size=3, hidden_size=8, num_layers=1, dense_size=4)
    model = RULLSTM(config).eval()
    path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": config.to_dict()},
        path,
    )

    restored = load_lstm_checkpoint(path)
    values = torch.zeros((2, 4, 3))

    assert torch.allclose(model(values), restored(values))
