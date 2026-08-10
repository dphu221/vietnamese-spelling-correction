"""Hierarchical Transformer for Vietnamese spelling correction.

This is a faithful, runnable interpretation of the architecture in
"Hierarchical Transformer Encoders for Vietnamese Spelling Correction".

The paper specifies that character-level outputs are concatenated with word
embeddings, but does not specify (1) how character outputs are reduced to one
vector per whitespace token or (2) the projection dimensions.  This
implementation makes those choices explicit:

* Each whitespace token is encoded independently by the character Transformer.
* The masked mean of its character outputs is its character vector O_i.
* [word_embedding_i ; O_i] is projected to the word Transformer hidden size.
* The corrector output matrix is tied to the input word embedding matrix.

Expected input shapes:
    word_ids:            (batch, num_tokens)
    char_ids:            (batch, num_tokens, max_chars_per_token)
    attention_mask:      (batch, num_tokens), True for a real token
    char_attention_mask: (batch, num_tokens, max_chars_per_token), True for
                         a real character
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpellingCorrectionConfig:
    """Hyperparameters for :class:`HierarchicalSpellingCorrector`.

    The character encoder depth (4), character hidden size (256), word encoder
    depth (12), vocabulary sizes (60,000 words and 400 characters), and maximum
    token length (192) follow the paper where possible.  The paper writes word
    hidden size ``786``; this is likely a typo for the BERT-base size 768, which
    is used here.
    """

    word_vocab_size: int = 60_000
    char_vocab_size: int = 400
    max_tokens: int = 192
    max_chars_per_token: int = 32
    pad_word_id: int = 0
    pad_char_id: int = 0

    char_hidden_size: int = 256
    char_num_layers: int = 4
    char_num_heads: int = 8

    word_embedding_size: int = 512
    word_hidden_size: int = 768
    word_num_layers: int = 12
    word_num_heads: int = 12
    share_word_layer_weights: bool = False

    detector_hidden_size: int = 256
    dropout: float = 0.1


def compact_1m_config() -> SpellingCorrectionConfig:
    """Return a ~1.0M-parameter configuration for a 6 GB laptop GPU.

    The configuration preserves the paper's hierarchy while adapting it to a
    small parameter budget:

    * 9,500 word entries: enough for the roughly 7,184 Vietnamese syllables
      cited by the paper, plus special, punctuation, and frequent extra tokens.
    * a 2-layer, 64-dimensional character encoder;
    * a 128-dimensional word encoder unrolled four times with ALBERT-style
      shared block weights.

    With PyTorch's current ``TransformerEncoderLayer`` parameterization this
    has approximately 1,001,310 trainable parameters.  The count may vary
    slightly across framework versions.
    """
    return SpellingCorrectionConfig(
        word_vocab_size=9_500,
        char_vocab_size=400,
        max_tokens=192,
        max_chars_per_token=32,
        char_hidden_size=64,
        char_num_layers=2,
        char_num_heads=4,
        word_embedding_size=64,
        word_hidden_size=128,
        word_num_layers=4,
        word_num_heads=4,
        share_word_layer_weights=True,
        detector_hidden_size=64,
        dropout=0.1,
    )


@dataclass
class SpellingCorrectionOutput:
    """Outputs returned by the model.

    ``correction_logits[b, t]`` is a score over the word vocabulary for token
    ``t``. It is only applied at inference if ``detection_logits`` predicts the
    token is erroneous.
    """

    detection_logits: torch.Tensor
    correction_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    detection_loss: Optional[torch.Tensor] = None
    correction_loss: Optional[torch.Tensor] = None


class HierarchicalSpellingCorrector(nn.Module):
    """Character-to-word hierarchical Transformer spelling corrector.

    The central bridge from characters to words is:

    ``[b, ô, j] -> Character Transformer -> O_bôj``
    ``[E_bôj ; O_bôj] -> token representation for the Word Transformer``

    Character attention is restricted to one whitespace-delimited token. Word
    attention then operates across all resulting token representations in the
    sentence, bidirectionally and in parallel.
    """

    def __init__(self, config: SpellingCorrectionConfig) -> None:
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(
            config.word_vocab_size,
            config.word_embedding_size,
            padding_idx=config.pad_word_id,
        )
        self.word_positions = nn.Embedding(config.max_tokens, config.word_hidden_size)

        self.char_embeddings = nn.Embedding(
            config.char_vocab_size,
            config.char_hidden_size,
            padding_idx=config.pad_char_id,
        )
        self.char_positions = nn.Embedding(
            config.max_chars_per_token,
            config.char_hidden_size,
        )

        char_layer = nn.TransformerEncoderLayer(
            d_model=config.char_hidden_size,
            nhead=config.char_num_heads,
            dim_feedforward=config.char_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.char_encoder = nn.TransformerEncoder(char_layer, config.char_num_layers)

        # Paper says to concatenate word and character features. This linear
        # layer is the explicit projection missing from its implementation detail.
        self.fusion = nn.Sequential(
            nn.Linear(config.word_embedding_size + config.char_hidden_size,
                      config.word_hidden_size),
            nn.LayerNorm(config.word_hidden_size),
            nn.Dropout(config.dropout),
        )

        word_layer = nn.TransformerEncoderLayer(
            d_model=config.word_hidden_size,
            nhead=config.word_num_heads,
            dim_feedforward=config.word_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # ALBERT-style sharing makes a small model deeper without multiplying
        # the parameter count. With sharing enabled, one physical block is run
        # ``word_num_layers`` times; otherwise this is a normal Transformer.
        self.shared_word_layer: Optional[nn.TransformerEncoderLayer]
        if config.share_word_layer_weights:
            self.shared_word_layer = word_layer
            self.word_encoder = None
        else:
            self.shared_word_layer = None
            self.word_encoder = nn.TransformerEncoder(word_layer, config.word_num_layers)

        self.detector = nn.Sequential(
            nn.Linear(config.word_hidden_size, config.detector_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.detector_hidden_size, 2),  # 0 = correct, 1 = error
        )

        # Project to word-embedding space, then use the input word embedding
        # matrix as the corrector classifier matrix (weight tying).
        self.corrector_projection = nn.Linear(
            config.word_hidden_size, config.word_embedding_size, bias=False
        )
        self.corrector_bias = nn.Parameter(torch.zeros(config.word_vocab_size))

    def encode_characters(
        self,
        char_ids: torch.Tensor,
        char_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one character-aware vector for every whitespace token.

        ``char_ids`` has shape ``(B, T, C)``. It is reshaped to ``(B*T, C)`` so
        the character encoder never attends across token boundaries. The masked
        mean then converts the C character outputs back to exactly one vector
        ``O_i`` per token.
        """
        batch_size, num_tokens, num_chars = char_ids.shape
        if num_chars > self.config.max_chars_per_token:
            raise ValueError(
                f"Got {num_chars} characters/token, but max is "
                f"{self.config.max_chars_per_token}."
            )

        flat_ids = char_ids.reshape(batch_size * num_tokens, num_chars)
        flat_mask = char_attention_mask.reshape(batch_size * num_tokens, num_chars).bool()

        # Padded word positions have no real character. Transformer attention
        # cannot receive an all-True key-padding mask (its softmax is undefined),
        # so expose one harmless PAD position internally. The original mask is
        # still used for pooling, therefore their resulting character vector is 0.
        safe_mask = flat_mask.clone()
        no_chars = ~safe_mask.any(dim=1)
        safe_mask[:, 0] |= no_chars

        positions = torch.arange(num_chars, device=char_ids.device)
        char_states = self.char_embeddings(flat_ids) + self.char_positions(positions)
        char_states = self.char_encoder(
            char_states,
            src_key_padding_mask=~safe_mask,
        )

        # The masked mean is the explicit token-level summary O_i.
        weights = flat_mask.unsqueeze(-1).to(char_states.dtype)
        summaries = (char_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return summaries.reshape(batch_size, num_tokens, self.config.char_hidden_size)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
        detection_labels: Optional[torch.Tensor] = None,
        correction_labels: Optional[torch.Tensor] = None,
    ) -> SpellingCorrectionOutput:
        """Compute detector and corrector logits, optionally with training loss.

        ``detection_labels`` uses 0 for correct tokens and 1 for erroneous
        tokens. ``correction_labels`` contains the intended word id. The
        correction loss is calculated only where the ground-truth detector label
        is 1, matching the paper's objective.
        """
        if word_ids.ndim != 2 or char_ids.ndim != 3:
            raise ValueError("word_ids must be (B, T), and char_ids must be (B, T, C).")
        if word_ids.shape[:2] != char_ids.shape[:2]:
            raise ValueError("word_ids and char_ids must agree on batch and token dimensions.")

        batch_size, num_tokens = word_ids.shape
        if num_tokens > self.config.max_tokens:
            raise ValueError(
                f"Got {num_tokens} tokens, but max_tokens is {self.config.max_tokens}."
            )
        if attention_mask is None:
            attention_mask = word_ids.ne(self.config.pad_word_id)
        else:
            attention_mask = attention_mask.bool()
        if char_attention_mask is None:
            char_attention_mask = char_ids.ne(self.config.pad_char_id)
        else:
            char_attention_mask = char_attention_mask.bool()

        char_vectors = self.encode_characters(char_ids, char_attention_mask)
        word_vectors = self.word_embeddings(word_ids)
        fused = self.fusion(torch.cat([word_vectors, char_vectors], dim=-1))

        token_positions = torch.arange(num_tokens, device=word_ids.device)
        contextual_vectors = fused + self.word_positions(token_positions)
        if self.shared_word_layer is not None:
            for _ in range(self.config.word_num_layers):
                contextual_vectors = self.shared_word_layer(
                    contextual_vectors,
                    src_key_padding_mask=~attention_mask,
                )
        else:
            assert self.word_encoder is not None
            contextual_vectors = self.word_encoder(
                contextual_vectors,
                src_key_padding_mask=~attention_mask,
            )

        detection_logits = self.detector(contextual_vectors)
        correction_space = self.corrector_projection(contextual_vectors)
        correction_logits = F.linear(
            correction_space,
            self.word_embeddings.weight,
            self.corrector_bias,
        )

        detection_loss = None
        correction_loss = None
        total_loss = None
        if detection_labels is not None:
            valid_detection = attention_mask & detection_labels.ge(0)
            if valid_detection.any():
                detection_loss = F.cross_entropy(
                    detection_logits[valid_detection], detection_labels[valid_detection]
                )
            else:
                detection_loss = detection_logits.sum() * 0.0

        if correction_labels is not None:
            if detection_labels is None:
                raise ValueError("correction_labels requires detection_labels to mask true errors.")
            valid_correction = (
                attention_mask
                & detection_labels.eq(1)
                & correction_labels.ge(0)
            )
            if valid_correction.any():
                correction_loss = F.cross_entropy(
                    correction_logits[valid_correction], correction_labels[valid_correction]
                )
            else:
                correction_loss = correction_logits.sum() * 0.0

        if detection_loss is not None and correction_loss is not None:
            total_loss = detection_loss + correction_loss
        elif detection_loss is not None:
            total_loss = detection_loss
        elif correction_loss is not None:
            total_loss = correction_loss

        return SpellingCorrectionOutput(
            detection_logits=detection_logits,
            correction_logits=correction_logits,
            loss=total_loss,
            detection_loss=detection_loss,
            correction_loss=correction_loss,
        )

    @torch.no_grad()
    def correct(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        char_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return corrected ids, error flags, and suggested replacement ids.

        This is one parallel forward pass. A proposed replacement is used only
        for positions predicted erroneous by the detector; it is not fed back
        into the model to re-correct the rest of the sentence.
        """
        output = self(
            word_ids=word_ids,
            char_ids=char_ids,
            attention_mask=attention_mask,
            char_attention_mask=char_attention_mask,
        )
        if attention_mask is None:
            attention_mask = word_ids.ne(self.config.pad_word_id)
        else:
            attention_mask = attention_mask.bool()

        error_flags = output.detection_logits.argmax(dim=-1).eq(1) & attention_mask
        suggestions = output.correction_logits.argmax(dim=-1)
        corrected_ids = torch.where(error_flags, suggestions, word_ids)
        return corrected_ids, error_flags, suggestions


if __name__ == "__main__":
    # Smoke test with a deliberately small model. Replace ids with values from
    # your real word/character vocabularies in training or inference code.
    tiny = SpellingCorrectionConfig(
        word_vocab_size=100,
        char_vocab_size=50,
        max_tokens=8,
        max_chars_per_token=6,
        char_hidden_size=32,
        char_num_heads=4,
        char_num_layers=1,
        word_embedding_size=48,
        word_hidden_size=48,
        word_num_heads=4,
        word_num_layers=1,
        detector_hidden_size=24,
        dropout=0.0,
    )
    model = HierarchicalSpellingCorrector(tiny)
    word_ids = torch.tensor([[5, 6, 7, 8]])
    char_ids = torch.tensor([[[2, 3, 4], [5, 6, 0], [7, 8, 9], [10, 0, 0]]])
    attention_mask = torch.tensor([[True, True, True, True]])
    char_mask = char_ids.ne(tiny.pad_char_id)
    output = model(word_ids, char_ids, attention_mask, char_mask)
    print("detector logits:", output.detection_logits.shape)  # (1, 4, 2)
    print("corrector logits:", output.correction_logits.shape)  # (1, 4, 100)
