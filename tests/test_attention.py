import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.layers import ScaledDotProductAttention, MultiHeadAttention

BATCH   = 2
HEADS   = 8
SEQ     = 10
D_MODEL = 64
D_K     = D_MODEL // HEADS   # 8


# ── ScaleDotProductAttention ──────────────────────────────────────────────────

class TestScaleDotProduct:

    def setup_method(self):
        self.attn = ScaledDotProductAttention()
        self.Q = torch.randn(BATCH, HEADS, SEQ, D_K)
        self.K = torch.randn(BATCH, HEADS, SEQ, D_K)
        self.V = torch.randn(BATCH, HEADS, SEQ, D_K)

    def test_output_shape(self):
        out, weights = self.attn(self.Q, self.K, self.V)
        assert out.shape     == (BATCH, HEADS, SEQ, D_K)
        assert weights.shape == (BATCH, HEADS, SEQ, SEQ)

    def test_weights_sum_to_one(self):
        _, weights = self.attn(self.Q, self.K, self.V)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weights_are_non_negative(self):
        _, weights = self.attn(self.Q, self.K, self.V)
        assert weights.min().item() >= 0.0

    def test_padding_mask_zeroes_out_weights(self):
        # mask position 0 — its weight must be 0 after softmax
        mask = torch.zeros(BATCH, 1, 1, SEQ, dtype=torch.bool)
        mask[:, :, :, 0] = True
        _, weights = self.attn(self.Q, self.K, self.V, mask=mask)
        assert weights[:, :, :, 0].abs().max().item() < 1e-6

    def test_causal_mask_upper_triangle_is_zero(self):
        # look-ahead mask: position i cannot attend to j > i
        mask = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)   # (1,1,seq,seq)
        _, weights = self.attn(self.Q, self.K, self.V, mask=mask)
        # upper triangle (excluding diagonal) must be zero
        for i in range(SEQ):
            for j in range(i + 1, SEQ):
                assert weights[:, :, i, j].abs().max().item() < 1e-6


# ── MultiHeadAttention ────────────────────────────────────────────────────────

class TestMultiHeadAttention:

    def setup_method(self):
        self.mha = MultiHeadAttention(embedding_dim=D_MODEL, num_heads=HEADS)
        self.x   = torch.randn(BATCH, SEQ, D_MODEL)

    def test_output_shape_self_attention(self):
        out = self.mha(self.x, self.x, self.x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_output_shape_cross_attention(self):
        # encoder output as K and V, decoder query as Q
        enc = torch.randn(BATCH, 15, D_MODEL)   # different seq length
        dec = torch.randn(BATCH, SEQ, D_MODEL)
        out = self.mha(dec, enc, enc)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_embedding_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(embedding_dim=65, num_heads=8)

    def test_weights_stored_after_forward(self):
        self.mha(self.x, self.x, self.x)
        assert hasattr(self.mha, "attention_weights")
        assert self.mha.attention_weights.shape == (BATCH, HEADS, SEQ, SEQ)

    def test_mask_applied_correctly(self):
        # padding mask on last position
        mask = torch.zeros(BATCH, 1, 1, SEQ, dtype=torch.bool)
        mask[:, :, :, -1] = True
        self.mha(self.x, self.x, self.x, mask=mask)
        assert self.mha.attention_weights[:, :, :, -1].abs().max().item() < 1e-6


    def test_different_q_k_produce_different_outputs(self):
        out1 = self.mha(self.x, self.x, self.x)
        q2   = torch.randn(BATCH, SEQ, D_MODEL)
        out2 = self.mha(q2, self.x, self.x)
        assert not torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
