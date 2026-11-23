"""A small 1-D convolutional network on the raw arterial waveform.

The network sees the 30-second pressure waveform (50 Hz, 1,500 samples) and learns
its own features, in contrast to the hand-crafted features in ``features.py``.
It is deliberately small (about 60,000 weights) so that it trains on a laptop CPU.

Input scaling: pressure in mmHg is mapped to roughly [-1, 1] with fixed constants
((x - 80) / 40). Fixed constants are used instead of per-window normalisation because
the absolute pressure level carries most of the information.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def _block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(4))


class WaveCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(_block(1, 16, 15), _block(16, 32, 9), _block(32, 64, 5), _block(64, 64, 5))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (batch, 1500)
        return self.head(self.features(x.unsqueeze(1))).squeeze(1)


def _scale(w: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((w.astype(np.float32) - 80.0) / 40.0)


@torch.no_grad()
def predict(model: WaveCNN, w: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(_scale(w[i:i + batch]))).numpy() for i in range(0, len(w), batch)]
    return np.concatenate(out)


def train(w_tr, y_tr, w_va, y_va, epochs: int = 8, batch: int = 256, lr: float = 2e-3, seed: int = 0,
          threads: int = 4) -> tuple[WaveCNN, list[dict]]:
    """Train with class-weighted loss; keep the weights of the epoch with the best validation AUROC."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    model = WaveCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor(float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x, y = _scale(w_tr), torch.from_numpy(y_tr.astype(np.float32))
    history, best_auc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward(); opt.step()
            total += loss.item() * len(idx)
        auc = float(roc_auc_score(y_va, predict(model, w_va)))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_auroc": auc})
        print(f"  cnn epoch {epoch}: loss {total / len(x):.4f}, validation AUROC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history
