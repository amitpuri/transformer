import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.model import Transformer

BATCH      = 2
SRC_SEQ    = 8
TGT_SEQ    = 6
SRC_VOCAB  = 100
TGT_VOCAB  = 120
D_MODEL    = 64
N_LAYERS   = 6
N_HEADS    = 8
D_FF       = 256
PAD_IDX    = 0


@pytest.fixture(scope="module")
def model():
    m = Transformer(
        source_vocab_size=SRC_VOCAB, 
        target_vocab_size=TGT_VOCAB, 
        embedding_dim=D_MODEL, 
        num_layers=N_LAYERS, 
        num_heads=N_HEADS, 
        feed_forward_dim=D_FF, 
        dropout=0.0
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def src():
    return torch.randint(1, SRC_VOCAB, (BATCH, SRC_SEQ))


@pytest.fixture(scope="module")
def tgt():
    return torch.randint(1, TGT_VOCAB, (BATCH, TGT_SEQ))


# ── Output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:

    def test_logits_shape(self, model, src, tgt):
        with torch.no_grad():
            logits = model(src, tgt)
        assert logits.shape == (BATCH, TGT_SEQ, TGT_VOCAB)

    def test_logits_are_raw_scores(self, model, src, tgt):
        # Raw logits — values outside [0,1] confirm no softmax applied
        with torch.no_grad():
            logits = model(src, tgt)
        assert logits.min().item() < 0.0 or logits.max().item() > 1.0


# ── Masks ─────────────────────────────────────────────────────────────────────

class TestMasks:

    def test_masks_built_automatically(self, model, src, tgt):
        # Should not raise even without explicit masks
        with torch.no_grad():
            out = model(src, tgt)
        assert out.shape == (BATCH, TGT_SEQ, TGT_VOCAB)

    def test_padding_in_src_does_not_crash(self, model, tgt):
        src_pad = torch.randint(1, SRC_VOCAB, (BATCH, SRC_SEQ))
        src_pad[:, -2:] = PAD_IDX
        with torch.no_grad():
            out = model(src_pad, tgt)
        assert out.shape == (BATCH, TGT_SEQ, TGT_VOCAB)


# ── Weight tying ──────────────────────────────────────────────────────────────

class TestWeightTying:

    def test_projection_shares_embedding_weights(self, model):
        # Named projection -> output_projection
        assert model.output_projection.weight is model.decoder.embedding.token_emb.weight


# ── Encode / decode_step helpers ──────────────────────────────────────────────

class TestInferenceHelpers:

    def test_encode_output_shape(self, model, src):
        with torch.no_grad():
            enc_out, src_mask = model.encode(src)
        assert enc_out.shape  == (BATCH, SRC_SEQ, D_MODEL)
        assert src_mask.shape == (BATCH, 1, 1, SRC_SEQ)

    def test_decode_step_output_shape(self, model, src, tgt):
        with torch.no_grad():
            enc_out, src_mask = model.encode(src)
            logits = model.decode_step(tgt, enc_out, src_mask)
        assert logits.shape == (BATCH, TGT_SEQ, TGT_VOCAB)

    def test_encode_once_decode_multiple(self, model, src):
        # Encode once, decode two different targets — enc_out must not change
        with torch.no_grad():
            enc_out, src_mask = model.encode(src)
            tgt_a = torch.randint(1, TGT_VOCAB, (BATCH, TGT_SEQ))
            tgt_b = torch.randint(1, TGT_VOCAB, (BATCH, TGT_SEQ))
            out_a = model.decode_step(tgt_a, enc_out, src_mask)
            out_b = model.decode_step(tgt_b, enc_out, src_mask)
        assert out_a.shape == out_b.shape
        assert not torch.allclose(out_a, out_b)



# ── Parameter count ───────────────────────────────────────────────────────────

class TestParameters:

    def test_has_parameters(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_parameter_count_printed(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        assert trainable == total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
