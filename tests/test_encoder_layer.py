import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.layers import LayerNormalization, PositionWiseFeedForward
from transformer.encoder import EncoderLayer

BATCH   = 2
SEQ     = 10
D_MODEL = 64
N_HEADS = 8
D_FF    = 256   # 4 × d_model


# ── LayerNormalization ────────────────────────────────────────────────────────

class TestLayerNorm:

    def setup_method(self):
        self.ln = LayerNormalization(D_MODEL)
        self.x  = torch.randn(BATCH, SEQ, D_MODEL)

    def test_output_shape(self):
        assert self.ln(self.x).shape == (BATCH, SEQ, D_MODEL)

    def test_mean_near_zero(self):
        out = self.ln(self.x)
        means = out.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_std_near_one(self):
        out = self.ln(self.x)
        stds = out.std(dim=-1, unbiased=False)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-5)

    def test_alpha_bias_are_learnable(self):
        param_names = [n for n, _ in self.ln.named_parameters()]
        assert "alpha" in param_names
        assert "bias"  in param_names

    def test_alpha_init_ones_bias_init_zeros(self):
        assert torch.all(self.ln.alpha == 1.0)
        assert torch.all(self.ln.bias  == 0.0)


# ── PositionWiseFeedForward ───────────────────────────────────────────────────

class TestFeedForward:

    def setup_method(self):
        self.ff = PositionWiseFeedForward(D_MODEL, D_FF, dropout=0.0)
        self.x  = torch.randn(BATCH, SEQ, D_MODEL)

    def test_output_shape(self):
        assert self.ff(self.x).shape == (BATCH, SEQ, D_MODEL)

    def test_different_positions_independent(self):
        # FFN is position-wise: changing one position must not affect others
        x2 = self.x.clone()
        x2[:, 0, :] = torch.randn(BATCH, D_MODEL)   # change position 0
        out1 = self.ff(self.x)
        out2 = self.ff(x2)
        # positions 1+ must be identical
        assert torch.allclose(out1[:, 1:, :], out2[:, 1:, :], atol=1e-6)


# ── EncoderLayer ──────────────────────────────────────────────────────────────

class TestEncoderLayer:

    def setup_method(self):
        self.layer = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        self.x     = torch.randn(BATCH, SEQ, D_MODEL)

    def test_output_shape(self):
        out = self.layer(self.x)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_output_is_normalised(self):
        self.layer.eval()
        out   = self.layer(self.x)
        means = out.mean(dim=-1)
        stds  = out.std(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
        assert torch.allclose(stds,  torch.ones_like(stds),  atol=1e-5)

    def test_residual_changes_output(self):
        # If residual connection is working, output != raw attention output
        self.layer.eval()
        with torch.no_grad():
            out      = self.layer(self.x)
            attn_raw = self.layer.self_attention(self.x, self.x, self.x)
        assert not torch.allclose(out, attn_raw)

    def test_padding_mask_applied(self):
        # Mask last position — its weights in attention must be zero
        mask = torch.zeros(BATCH, 1, 1, SEQ, dtype=torch.bool)
        mask[:, :, :, -1] = True
        self.layer(self.x, mask=mask)
        weights = self.layer.self_attention.attention_weights
        assert weights[:, :, :, -1].abs().max().item() < 1e-6

    def test_stacking_two_layers_shape(self):
        layer2 = EncoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        out = layer2(self.layer(self.x))
        assert out.shape == (BATCH, SEQ, D_MODEL)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
