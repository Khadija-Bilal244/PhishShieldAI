"""Knowledge distillation + plain training loops for the CharCNN student."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from phishshield.features import char_tokenizer
from phishshield.models.student import CharCNNStudent
from phishshield.training.common import Split


def fit_student(model: CharCNNStudent, train: Split, cfg: dict, teacher_logits: np.ndarray | None = None,
                epochs: int | None = None, lr: float | None = None, partial: bool = False, seed: int = 0) -> dict:
    """Train in place. If teacher_logits is None this is ordinary supervised training."""
    torch.manual_seed(seed)
    if partial:
        model.freeze_for_partial()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr or cfg["lr"])
    T, alpha = cfg["temperature"], cfg["alpha"]
    ids_all = torch.from_numpy(char_tokenizer.encode_batch(train.urls, cfg["max_len"]))
    feats_all = torch.as_tensor(np.array(train.Xs, dtype=np.float32))
    y_all = torch.as_tensor(np.array(train.y, dtype=np.float32))
    t_all = torch.as_tensor(teacher_logits, dtype=torch.float32) if teacher_logits is not None else None
    pos = float(y_all.mean().clamp(1e-3, 1 - 1e-3))
    pos_weight = torch.tensor((1 - pos) / pos)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs or cfg["epochs"]):
        order = rng.permutation(len(train))
        for i in range(0, len(order), cfg["batch_size"]):
            idx = torch.from_numpy(order[i:i + cfg["batch_size"]])
            logit, _ = model(ids_all[idx], feats_all[idx])
            hard = F.binary_cross_entropy_with_logits(logit, y_all[idx], pos_weight=pos_weight)
            if t_all is not None:
                soft_t = torch.sigmoid(t_all[idx] / T)
                soft = F.binary_cross_entropy_with_logits(logit / T, soft_t) * T * T
                loss = alpha * soft + (1 - alpha) * hard
            else:
                loss = hard
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.unfreeze_all()
    model.eval()
    return {"train_time_s": time.perf_counter() - t0}
