"""Training loop for the hybrid DistilBERT fusion model (used by train, ablation and updates)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from phishshield.evaluation.metrics import classification_metrics
from phishshield.models.hybrid import HybridPhishModel, TextEncoder, predict_logits
from phishshield.training.common import Split, device, sigmoid
from phishshield.utils import tracking
from phishshield.utils.common import get_logger

log = get_logger("hybrid_trainer")


def build_hybrid(cfg: dict, n_features: int, mode: str | None = None) -> HybridPhishModel:
    enc = TextEncoder(cfg["pretrained_name"], cfg["debug_tiny"], cfg["max_len"])
    return HybridPhishModel(enc, n_features, cfg["fusion_dim"], cfg["dropout"], mode or cfg["mode"])


def fit_hybrid(model: HybridPhishModel, train: Split, val: Split, cfg: dict, epochs: int | None = None,
               partial: bool = False, log_prefix: str = "") -> dict:
    """Train in place; keeps the best epoch by validation F1. Returns training summary."""
    dev = device()
    model.to(dev)
    if partial:
        model.freeze_lower_layers(cfg["freeze_layers_partial"])
    epochs = epochs or cfg["epochs"]

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr"]},
                             {"params": head_params, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    pos = max(train.y.mean(), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    steps_total = epochs * int(np.ceil(len(train) / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lr"], cfg["head_lr"]], total_steps=max(steps_total, 1),
                                                pct_start=0.1)

    best_f1, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            tok = model.encoder_wrapper.tokenize([train.urls[j] for j in idx], dev)
            f = torch.as_tensor(train.Xs[idx], device=dev)
            y = torch.as_tensor(train.y[idx], dtype=torch.float32, device=dev)
            loss = loss_fn(model(tok["input_ids"], tok["attention_mask"], f), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val_m = classification_metrics(val.y, sigmoid(predict_logits(model, val.urls, val.Xs, device=dev)))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_m.items()}})
        tracking.log_metrics({"train_loss": float(np.mean(losses)), "val_f1": val_m["f1"],
                              "val_recall": val_m["recall"], "val_fnr": val_m["fnr"]}, prefix=log_prefix, step=epoch)
        log.info("epoch %d loss=%.4f val_f1=%.4f val_fnr=%.4f", epoch, np.mean(losses), val_m["f1"], val_m["fnr"])
        if val_m["f1"] > best_f1:
            best_f1, best_state = val_m["f1"], copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.unfreeze_all()
    model.to("cpu").eval()
    return {"best_val_f1": best_f1, "train_time_s": time.perf_counter() - t0, "history": history}
