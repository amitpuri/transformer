import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder, DecoderLayer

BATCH    = 2
SRC_SEQ  = 8
TGT_SEQ  = 6
VOCAB    = 100
D_MODEL  = 64
N_LAYERS = 6
N_HEADS  = 8
D_FF     = 256
PAD_IDX  = 0


@pytest.fixture(scope="module")
def enc_output():
    encoder = TransformerEncoder(
        vocab_size=VOCAB, 
        embedding_dim=D_MODEL, 
        num_layers=N_LAYERS, 
        num_heads=N_HEADS, 
        feed_forward_dim=D_FF, 
        dropout=0.0
    )
    encoder.eval()
    src = torch.randint(1, VOCAB, (BATCH, SRC_SEQ))
    with torch.no_grad():
        return encoder(src)


@pytest.fixture(scope="module")
def decoder():
    return TransformerDecoder(
        vocab_size=VOCAB, 
        embedding_dim=D_MODEL, 
        num_layers=N_LAYERS, 
        num_heads=N_HEADS, 
        feed_forward_dim=D_FF, 
        dropout=0.0
    )


# ── DecoderLayer ──────────────────────────────────────────────────────────────

class TestDecoderLayer:

    def test_output_shape(self, enc_output):
        layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        tgt   = torch.randn(BATCH, TGT_SEQ, D_MODEL)
        out   = layer(tgt, enc_output)
        assert out.shape == (BATCH, TGT_SEQ, D_MODEL)

    def test_three_sublayers_present(self):
        layer = DecoderLayer(D_MODEL, N_HEADS, D_FF)
        assert hasattr(layer, "self_attention")
        assert hasattr(layer, "cross_attention")
        assert hasattr(layer, "feed_forward")

    def test_cross_attention_uses_encoder_output(self, enc_output):
        layer = DecoderLayer(D_MODEL, N_HEADS, D_FF, dropout=0.0)
        tgt   = torch.randn(BATCH, TGT_SEQ, D_MODEL)
        enc2  = torch.randn_like(enc_output)
        out1  = layer(tgt, enc_output)
        out2  = layer(tgt, enc2)
        # Different encoder output → different decoder output
        assert not torch.allclose(out1, out2)


# ── Decoder ───────────────────────────────────────────────────────────────────

class TestDecoder:

    def test_output_shape(self, decoder, enc_output):
        tgt = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        out = decoder(tgt, enc_output)
        assert out.shape == (BATCH, TGT_SEQ, D_MODEL)

    def test_num_layers(self, decoder):
        assert len(decoder.layers) == N_LAYERS

    # ── Causal mask ──────────────────────────────────────────────────────────

    def test_causal_mask_shape(self, decoder):
        tgt  = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        mask = decoder.build_target_mask(tgt, PAD_IDX)
        assert mask.shape == (BATCH, 1, TGT_SEQ, TGT_SEQ)

    def test_causal_mask_upper_triangle_blocked(self, decoder):
        tgt  = torch.randint(1, VOCAB, (1, TGT_SEQ))
        mask = decoder.build_target_mask(tgt, PAD_IDX)
        # upper triangle (diagonal=1) must all be True (blocked)
        for i in range(TGT_SEQ):
            for j in range(i + 1, TGT_SEQ):
                assert mask[0, 0, i, j].item() == True

    def test_causal_mask_diagonal_and_below_not_blocked(self, decoder):
        tgt  = torch.randint(1, VOCAB, (1, TGT_SEQ))
        mask = decoder.build_target_mask(tgt, PAD_IDX)
        # lower triangle + diagonal must all be False (visible)
        for i in range(TGT_SEQ):
            for j in range(0, i + 1):
                assert mask[0, 0, i, j].item() == False

    def test_padding_positions_blocked_in_mask(self, decoder):
        tgt = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        tgt[0, -2:] = PAD_IDX
        mask = decoder.build_target_mask(tgt, PAD_IDX)
        # padding positions in last 2 cols must be True for row 0
        assert mask[0, 0, 0, -1].item() == True
        assert mask[0, 0, 0, -2].item() == True

    # ── Integration ──────────────────────────────────────────────────────────

    def test_full_encoder_decoder_pipeline(self, enc_output):
        decoder = TransformerDecoder(
            vocab_size=VOCAB, 
            embedding_dim=D_MODEL, 
            num_layers=N_LAYERS, 
            num_heads=N_HEADS, 
            feed_forward_dim=D_FF, 
            dropout=0.0
        )
        decoder.eval()
        tgt      = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        tgt_mask = decoder.build_target_mask(tgt, PAD_IDX)
        with torch.no_grad():
            out = decoder(tgt, enc_output, target_mask=tgt_mask)
        assert out.shape == (BATCH, TGT_SEQ, D_MODEL)

    def test_different_tgt_produces_different_output(self, decoder, enc_output):
        decoder.eval()
        tgt1 = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        tgt2 = torch.randint(1, VOCAB, (BATCH, TGT_SEQ))
        with torch.no_grad():
            out1 = decoder(tgt1, enc_output)
            out2 = decoder(tgt2, enc_output)
        assert not torch.allclose(out1, out2)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
