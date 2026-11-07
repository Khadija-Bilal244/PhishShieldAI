"""PRIMARY MODEL - Hybrid DistilBERT + engineered-feature fusion network.

            URL text ──> DistilBERT ──> [CLS] ──> Linear ──> h_t ─┐
                                                                  ├─> gate g = σ(W[h_t; h_f])
   engineered features ──> MLP ─────────────────────────> h_f ───┘     z = g·h_t + (1-g)·h_f
                                                                         │
                                                                    classifier ──> logit

`mode` supports the ablations required in the proposal:
    fusion        - full model (default)
    text_only     - transformer branch only
    features_only - feature branch only
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phishshield.features import char_tokenizer


class TextEncoder:
    """Wraps tokenizer + transformer so real and tiny (offline) encoders share one interface."""

    def __init__(self, pretrained_name: str, debug_tiny: bool, max_len: int, local_dir: str | None = None):
        """local_dir: a saved `encoder/` folder (tokenizer + config). When given, the architecture is
        rebuilt from the config without downloading anything; weights are then loaded from model.pt."""
        from transformers import AutoConfig, AutoModel, AutoTokenizer, DistilBertConfig, DistilBertModel

        self.debug_tiny = debug_tiny
        self.max_len = max_len
        self.pretrained_name = pretrained_name
        if debug_tiny:
            cfg = DistilBertConfig(vocab_size=char_tokenizer.VOCAB_SIZE, dim=64, n_layers=2, n_heads=2,
                                   hidden_dim=128, max_position_embeddings=max(max_len, 64), pad_token_id=0)
            self.model = DistilBertModel(cfg)
            self.tokenizer = None
        elif local_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(local_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
            self.model = AutoModel.from_pretrained(pretrained_name)
        self.hidden_size = self.model.config.dim if hasattr(self.model.config, "dim") else self.model.config.hidden_size

    def tokenize(self, urls: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        if self.debug_tiny:
            ids = torch.from_numpy(char_tokenizer.encode_batch(urls, self.max_len))
            return {"input_ids": ids.to(device), "attention_mask": (ids != 0).long().to(device)}
        enc = self.tokenizer(list(urls), padding="max_length", truncation=True, max_length=self.max_len,
                             return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def save(self, out_dir: Path) -> None:
        if not self.debug_tiny:
            self.tokenizer.save_pretrained(out_dir / "encoder")
            self.model.config.save_pretrained(out_dir / "encoder")


class HybridPhishModel(nn.Module):
    def __init__(self, encoder: TextEncoder, n_features: int, fusion_dim: int = 128,
                 dropout: float = 0.1, mode: str = "fusion"):
        super().__init__()
        assert mode in {"fusion", "text_only", "features_only"}
        self.mode = mode
        self.encoder_wrapper = encoder
        self.encoder = encoder.model
        self.text_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.hidden_size, fusion_dim), nn.GELU())
        self.feat_mlp = nn.Sequential(nn.Linear(n_features, fusion_dim), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(fusion_dim, fusion_dim), nn.GELU())
        self.gate = nn.Linear(2 * fusion_dim, fusion_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, 1))

    def cls_embedding(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def fused(self, input_ids, attention_mask, features) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "features_only":
            return self.feat_mlp(features), None
        h_t = self.text_proj(self.cls_embedding(input_ids, attention_mask))
        if self.mode == "text_only":
            return h_t, None
        h_f = self.feat_mlp(features)
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_f], dim=-1)))
        return g * h_t + (1 - g) * h_f, g

    def forward(self, input_ids, attention_mask, features) -> torch.Tensor:
        z, _ = self.fused(input_ids, attention_mask, features)
        return self.head(z).squeeze(-1)

    # ---------- partial fine-tuning support (tier 2 of the update policy) ----------
    def freeze_lower_layers(self, n_frozen: int) -> None:
        """Freeze embeddings + the lowest `n_frozen` transformer layers."""
        for p in self.encoder.embeddings.parameters():
            p.requires_grad = False
        layers = self.encoder.transformer.layer
        for layer in layers[: min(n_frozen, len(layers))]:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def save_hybrid(model: HybridPhishModel, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.encoder_wrapper.save(out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_hybrid(out_dir: Path, map_location: str = "cpu") -> tuple[HybridPhishModel, dict]:
    with open(out_dir / "config.json") as f:
        meta = json.load(f)
    local = out_dir / "encoder"
    encoder = TextEncoder(meta["pretrained_name"], meta["debug_tiny"], meta["max_len"],
                          local_dir=str(local) if local.exists() else None)
    model = HybridPhishModel(encoder, meta["n_features"], meta["fusion_dim"], meta["dropout"], meta["mode"])
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=map_location))
    model.eval()
    return model, meta


@torch.no_grad()
def predict_logits(model: HybridPhishModel, urls, feats: np.ndarray, batch_size: int = 64,
                   device: torch.device | None = None, return_embeddings: bool = False):
    device = device or next(model.parameters()).device
    model.eval()
    logits, embs = [], []
    urls = list(urls)
    for i in range(0, len(urls), batch_size):
        tok = model.encoder_wrapper.tokenize(urls[i:i + batch_size], device)
        f = torch.as_tensor(feats[i:i + batch_size], dtype=torch.float32, device=device)
        logits.append(model(tok["input_ids"], tok["attention_mask"], f).cpu().numpy())
        if return_embeddings:
            embs.append(model.cls_embedding(tok["input_ids"], tok["attention_mask"]).cpu().numpy())
    logits = np.concatenate(logits) if logits else np.zeros(0)
    if return_embeddings:
        return logits, (np.concatenate(embs) if embs else np.zeros((0, 1)))
    return logits
