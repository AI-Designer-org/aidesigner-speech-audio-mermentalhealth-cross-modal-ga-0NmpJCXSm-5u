"""
Clinical Concept Bottleneck (CCB) for the MER Mental Health architecture.

Maps session-level representations through an interpretable clinical concept
space aligned to DSM-5 depression criteria (PHQ-8), producing:
  - PHQ-8 total score (regression)
  - Per-item PHQ-8 scores (8 items, each [0, 3])
  - Binary depression classification (PHQ-8 >= 10)
  - Concept activation scores (interpretable intermediate layer)

The concept bottleneck makes the model's reasoning transparent by showing
which clinical concepts drove the prediction, analogous to physics-informed
ML where the intermediate representation is constrained to clinically
meaningful variables.

Architecture:
  Session Feature (B, d_session)
    -> MLP -> Concept Scores (B, n_concepts)  [sigmoid, in [0,1]]
    -> Item Predictor (B, n_items)            [scaled sigmoid, in [0,3]]
    -> Total Score (B,)                        [sum of items or direct regression]
    -> Depression Prob (B,)                    [sigmoid binary classifier]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from layers import LayerNorm, FeedForward
from config import ModelConfig


# ---------------------------------------------------------------------------
# Base operator abstract class
# ---------------------------------------------------------------------------

class BaseConceptBottleneck(ABC, nn.Module):
    """Base class for the clinical concept bottleneck operator."""

    @abstractmethod
    def forward(self, session_feat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            session_feat: (B, d_session_model)
        Returns:
            Tuple of (phq8_total, phq8_items, concept_scores, depression_prob)
        """
        pass


# ---------------------------------------------------------------------------
# Clinical Concept Bottleneck
# ---------------------------------------------------------------------------

class ClinicalConceptBottleneck(BaseConceptBottleneck):
    """Clinical Concept Bottleneck (CCB).

    Maps session representations through a clinically-grounded concept space
    aligned to PHQ-8 items.

    The concept->item weight matrix W is initialized diagonal-dominant
    (each concept primarily influences the corresponding PHQ-8 item) to
    encourage interpretable mapping. An L1 sparsity penalty on off-diagonal
    elements further enforces clinical grounding.

    Shape flow:
        (B, d_session) -> (B, concept_hidden) -> (B, n_concepts)
        -> (B, n_items) -> (B,) total + (B,) depression_prob
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_session = config.d_session_model
        n_concepts = config.n_clinical_concepts     # 8 = aligned to PHQ-8
        n_items = config.n_phq8_items               # 8

        # ── Concept predictor MLP ──
        # session_feat -> hidden -> hidden -> concept_logits
        self.concept_predictor = nn.Sequential(
            LayerNorm(d_session),
            nn.Linear(d_session, config.concept_hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.concept_dropout),
            nn.Linear(config.concept_hidden_dim, config.concept_hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(config.concept_dropout),
            nn.Linear(config.concept_hidden_dim, n_concepts, bias=False),
        )

        # ── Concept -> Item weight matrix ──
        # W: (n_items, n_concepts) — maps concept scores to item logits
        # Initialized diagonal-dominant for interpretability
        self.concept_to_item = nn.Linear(n_concepts, n_items, bias=False)
        self._init_item_weights(n_concepts, n_items)

        # ── Direct regression head (bypass items) ──
        # Used only if config.item_prediction_head is False
        self.direct_regression = nn.Linear(d_session, 1, bias=False)

        # ── Binary classification head ──
        # Classifies from concept scores (clinical prior: depression = multiple concepts)
        self.depression_classifier = nn.Linear(n_concepts, 1, bias=False)

        # ── Per-item regression heads (alternative to matrix) ──
        # For finer-grained per-item prediction
        self.item_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_concepts, config.head_hidden_dim, bias=False),
                nn.GELU(approximate="tanh"),
                nn.Dropout(config.concept_dropout),
                nn.Linear(config.head_hidden_dim, 1, bias=False),
            ) for _ in range(n_items)
        ])

    def _init_item_weights(self, n_concepts: int, n_items: int):
        """Initialize concept->item weight matrix with diagonal dominance.

        Each concept k primarily maps to item k; off-diagonal weights
        are initialized small to encourage interpretable mapping.
        """
        with torch.no_grad():
            # Weight shape: (n_items, n_concepts)
            W = self.concept_to_item.weight
            nn.init.normal_(W, mean=0.0, std=0.01)
            # Set diagonal to larger value (each concept -> corresponding item)
            diag_idx = torch.arange(min(n_concepts, n_items))
            W[diag_idx, diag_idx] = 1.0

    def forward(self, session_feat: torch.Tensor,
                concept_labels: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            session_feat: (B, d_session_model) — session-level representation
            concept_labels: optional (B, n_concepts) — ground truth concept scores
                              for supervised concept training
        Returns:
            phq8_total:      (B,)   — predicted PHQ-8 total score [0, 24]
            phq8_items:      (B, 8) — per-item severity [0, 3] each
            concept_scores:  (B, n_clinical_concepts) — concept activations [0, 1]
            depression_prob: (B,)   — probability PHQ-8 >= 10
        """
        config = self.config
        B = session_feat.shape[0]

        # ── 1. Concept prediction ──
        concept_logits = self.concept_predictor(session_feat)           # (B, n_concepts)
        concept_scores = torch.sigmoid(concept_logits)                  # (B, n_concepts) in [0, 1]

        # ── 2. PHQ-8 item prediction ──
        if config.item_prediction_head:
            # Option A: Per-item heads (more expressive)
            item_scores = []
            for i in range(config.n_phq8_items):
                item_logit = self.item_heads[i](concept_scores)         # (B, 1)
                item_scores.append(item_logit)

            # Alternative: use concept_to_item matrix as well (blend both)
            matrix_items = self.concept_to_item(concept_scores)          # (B, 8)

            # Blend: 0.5 * per-item heads + 0.5 * matrix
            item_logits = 0.5 * torch.cat(item_scores, dim=-1) + 0.5 * matrix_items  # (B, 8)

            # Scale to [0, 3] per PHQ-8 item scoring
            phq8_items = 3.0 * torch.sigmoid(item_logits)               # (B, 8) in [0, 3]

            # PHQ-8 total = sum of items
            phq8_total = phq8_items.sum(dim=-1)                         # (B,) in [0, 24]
        else:
            # Direct regression (bypass concept bottleneck for total score)
            phq8_total = self.direct_regression(session_feat).squeeze(-1)  # (B,)
            phq8_items = torch.zeros(B, config.n_phq8_items, device=session_feat.device)

        # ── 3. Binary depression classification ──
        if config.classification_head:
            depression_logit = self.depression_classifier(concept_scores).squeeze(-1)  # (B,)
            depression_prob = torch.sigmoid(depression_logit)                         # (B,)
        else:
            depression_prob = torch.zeros(B, device=session_feat.device)

        # ── Safety ──
        if self.training:
            assert not torch.isnan(phq8_total).any(), "NaN in CCB phq8_total"
            assert not torch.isnan(concept_scores).any(), "NaN in CCB concept_scores"

        return phq8_total, phq8_items, concept_scores, depression_prob

    def get_concept_importance(self, session_feat: torch.Tensor) -> torch.Tensor:
        """Return concept scores for interpretability analysis.

        Args:
            session_feat: (B, d_session_model)
        Returns:
            concept_scores: (B, n_concepts) — concept activations [0, 1]
        """
        with torch.no_grad():
            concept_logits = self.concept_predictor(session_feat)
            concept_scores = torch.sigmoid(concept_logits)
        return concept_scores

    def get_concept_to_item_weights(self) -> torch.Tensor:
        """Return the concept->item weight matrix for interpretability.

        Returns:
            W: (n_items, n_concepts) — weight matrix
        """
        return self.concept_to_item.weight.detach()
