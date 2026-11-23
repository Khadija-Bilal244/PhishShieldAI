"""Distilled STUDENT model - compact character-level CNN + feature branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phishshield.features import char_tokenizer


class CharCNNStudent(nn.Module):
    def __init__(self, n_features: int, embed_dim: int = 32, num_filters: int = 64,
                 kernel_sizes: tuple[int, ...] = (3, 5, 7), dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(char_tokenizer.VOCAB_SIZE, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.feat = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU())
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_filters * len(kernel_sizes) + 64, 64), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def embedding(self, char_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = self.embed(char_ids).transpose(1, 2)
        pooled = [torch.amax(torch.relu(c(x)), dim=2) for c in self.convs]
        return self.proj(torch.cat(pooled + [self.feat(features)], dim=1))

    def forward(self, char_ids: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(char_ids, features)
        return self.head(emb).squeeze(-1), emb

    def freeze_for_partial(self) -> None:
        """Tier-2 update: freeze the lowest layer (character embeddings) and fine-tune the conv filters."""
        for p in self.embed.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def build_student(cfg: dict, n_features: int) -> CharCNNStudent:
    return CharCNNStudent(n_features, cfg["embed_dim"], cfg["num_filters"], tuple(cfg["kernel_sizes"]))


def save_student(model: CharCNNStudent, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "student.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_student(out_dir: Path) -> tuple[CharCNNStudent, dict]:
    with open(out_dir / "config.json") as f:
        meta = json.load(f)
    model = CharCNNStudent(meta["n_features"], meta["embed_dim"], meta["num_filters"], tuple(meta["kernel_sizes"]))
    model.load_state_dict(torch.load(out_dir / "student.pt", map_location="cpu"))
    model.eval()
    return model, meta


@torch.no_grad()
def student_logits(model: CharCNNStudent, urls, feats: np.ndarray, max_len: int,
                   batch_size: int = 512, return_embeddings: bool = False):
    model.eval()
    urls = list(urls)
    outs, embs = [], []
    for i in range(0, len(urls), batch_size):
        ids = torch.from_numpy(char_tokenizer.encode_batch(urls[i:i + batch_size], max_len))
        f = torch.as_tensor(feats[i:i + batch_size], dtype=torch.float32)
        logit, emb = model(ids, f)
        outs.append(logit.numpy())
        embs.append(emb.numpy())
    logits = np.concatenate(outs) if outs else np.zeros(0)
    if return_embeddings:
        return logits, (np.concatenate(embs) if embs else np.zeros((0, 64)))
    return logits
