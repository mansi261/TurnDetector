"""Whisper encoder with attention pooling for binary turn decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class WhisperTurnDetector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        model_config: dict[str, Any],
    ):
        super().__init__()
        self.encoder = encoder
        self.model_config = model_config
        hidden_size = int(encoder.config.d_model)
        attention_size = int(model_config["attention_hidden_size"])
        dropout = float(model_config["dropout"])
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )
        head_layers: list[nn.Module] = []
        input_size = hidden_size
        for output_size in model_config["head_hidden_sizes"]:
            head_layers.extend(
                [nn.Linear(input_size, int(output_size)), nn.GELU(), nn.Dropout(dropout)]
            )
            input_size = int(output_size)
        head_layers.append(nn.Linear(input_size, 1))
        self.classifier = nn.Sequential(*head_layers)

    @classmethod
    def from_backbone(cls, model_config: dict[str, Any]):
        from transformers import WhisperModel

        backbone = WhisperModel.from_pretrained(
            model_config["backbone"], revision=model_config.get("backbone_revision")
        )
        original_positions = backbone.encoder.embed_positions.weight.detach()[:400].clone()
        backbone.config.max_source_positions = 400
        backbone.encoder.embed_positions = nn.Embedding(400, backbone.config.d_model)
        backbone.encoder.embed_positions.weight = nn.Parameter(
            original_positions, requires_grad=False
        )
        return cls(backbone.encoder, model_config=model_config)

    @classmethod
    def from_config(cls, model_config: dict[str, Any], whisper_config: dict[str, Any]):
        from transformers import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        config = WhisperConfig.from_dict(whisper_config)
        return cls(WhisperEncoder(config), model_config=model_config)

    def freeze_encoder(self, frozen: bool = True) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_features=input_features).last_hidden_state
        weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=-1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(-1)

    def checkpoint_payload(
        self,
        *,
        threshold: float,
        training_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_config": self.model_config,
            "whisper_config": self.encoder.config.to_dict(),
            "state_dict": self.state_dict(),
            "threshold": threshold,
            "training_state": training_state,
        }


def load_model_checkpoint(path: str | Path, *, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = WhisperTurnDetector.from_config(
        payload["model_config"], payload["whisper_config"]
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
