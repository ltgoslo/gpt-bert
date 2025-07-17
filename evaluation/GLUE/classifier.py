from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.nonlinearity: nn.Sequential = nn.Sequential(
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(self, eembeddings):
        return self.nonlinearity(eembeddings)


def import_architecture(architecture):
    if architecture == "base":
        from model_final import Bert
    elif architecture == "no_denseformer":
        from model_denseformer_attention_gate_minus_denseformer import Bert
    elif architecture == "no_gate":
        from model_denseformer_minus_attention_gate import Bert

    return Bert


class ModelForSequenceClassification(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transformer = import_architecture(config.architecture)(config)
        self.classifier = ClassifierHead(config)

    def forward(self, input_data, attention_mask):
        head_embedding = self.transformer.get_contextualized(input_data.t(), attention_mask.unsqueeze(1))[0]
        logits = self.classifier(head_embedding)

        return logits
