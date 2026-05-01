"""
Tests for the Input Embedding block.

Covers:
  - TokenEmbedding output shape and padding behaviour
  - PositionalEncoding shape, non-learnability, and periodicity
  - TransformerEmbedding combined output shape and dropout behaviour
"""

import math
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.embeddings import TokenEmbedding, SinusoidalPositionalEncoding, TransformerEmbedding

VOCAB  = 1000
D      = 64
BATCH  = 4
SEQ    = 20
MAX_LEN = 512


# ─── TokenEmbedding ──────────────────────────────────────────────────────────

class TestTokenEmbedding:
    def setup_method(self):
        self.emb = TokenEmbedding(VOCAB, D)

    def test_output_shape(self):
        x = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = self.emb(x)
        assert out.shape == (BATCH, SEQ, D), f"Expected ({BATCH},{SEQ},{D}), got {out.shape}"

    def test_padding_is_zero(self):
        """Padding token (index 0) must produce an all-zero embedding."""
        pad = torch.zeros(1, 1, dtype=torch.long)
        out = self.emb(pad)
        assert out.abs().sum().item() == 0.0

    def test_weights_are_learnable(self):
        assert self.emb.weight.requires_grad


# ─── SinusoidalPositionalEncoding ─────────────────────────────────────────────

class TestSinusoidalPositionalEncoding:
    def setup_method(self):
        self.pe = SinusoidalPositionalEncoding(D, MAX_LEN)

    def test_output_shape(self):
        x = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = self.pe(x)
        assert out.shape == (SEQ, D), f"Expected ({SEQ},{D}), got {out.shape}"

    def test_not_a_parameter(self):
        """Encoding must be a buffer, not a learnable parameter."""
        param_names = [n for n, _ in self.pe.named_parameters()]
        assert "pe" not in param_names  # Buffer name is 'pe'

    def test_even_dims_are_sin(self):
        """Even-indexed dimensions should be in [-1, 1] and not identical across positions."""
        x = torch.zeros(1, MAX_LEN, dtype=torch.long)
        enc = self.pe(x)           # (MAX_LEN, D)
        even_col = enc[:, 0]       # first even column → sin
        # Must vary across positions (not constant)
        assert even_col.std().item() > 0.1

    def test_sin_cos_range(self):
        x = torch.zeros(1, MAX_LEN, dtype=torch.long)
        enc = self.pe(x)
        assert enc.min().item() >= -1.0 - 1e-5
        assert enc.max().item() <=  1.0 + 1e-5


# ─── TransformerEmbedding ────────────────────────────────────────────────────

class TestTransformerEmbedding:
    def setup_method(self):
        self.emb = TransformerEmbedding(VOCAB, D, MAX_LEN, dropout=0.0)

    def test_output_shape(self):
        x = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = self.emb(x)
        assert out.shape == (BATCH, SEQ, D)

    def test_scale_applied(self):
        """
        With positional encoding zeroed out at pos=0 (sin(0)=0, cos(0)=1 only for dim 1),
        we verify token embeddings are scaled by sqrt(embedding_dim).
        Compare raw token embedding vs transformer embedding (no dropout).
        """
        self.emb.eval()
        x = torch.randint(1, VOCAB, (1, 1))
        tok_raw = self.emb.token_emb(x)                   # (1, 1, D)
        tok_scaled = tok_raw * math.sqrt(D)
        pos = self.emb.pos_enc(x)                         # (1, D)
        expected = (tok_scaled + pos).squeeze()
        actual   = self.emb(x).squeeze()
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_different_positions_differ(self):
        """Two different positions must not produce identical embeddings."""
        self.emb.eval()
        x = torch.ones(1, 10, dtype=torch.long)           # same token, 10 positions
        out = self.emb(x)                                  # (1, 10, D)
        # pos 0 and pos 5 should differ
        assert not torch.allclose(out[0, 0], out[0, 5])

    def test_dropout_changes_output_in_training(self):
        """With high dropout, training output differs from eval output."""
        emb = TransformerEmbedding(VOCAB, D, MAX_LEN, dropout=0.9)
        x = torch.randint(1, VOCAB, (BATCH, SEQ))
        emb.train()
        out_train = emb(x)
        emb.eval()
        out_eval  = emb(x)
        # Very unlikely to be identical under 90 % dropout
        assert not torch.allclose(out_train, out_eval)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
