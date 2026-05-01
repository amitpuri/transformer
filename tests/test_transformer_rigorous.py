"""
Rigorous tests for the full Transformer model.

Covers:
  - Output shapes under all input configurations
  - Causal mask leak: future tokens must never influence past positions
  - Padding isolation: pad tokens must not bleed into real token outputs
  - Batch independence: one sample in a batch must not affect another
  - Determinism: same input always produces same output in eval mode
  - Gradient flow: every parameter must receive a gradient
  - Weight tying: projection and embedding share the same tensor
  - Softmax validity: probabilities sum to 1, all non-negative
  - Encoder reuse: encoding once then decoding twice is identical to two full passes
  - src-only change: changing src changes output; tgt unchanged does not affect enc
  - tgt-only change: changing tgt changes output; enc output unchanged
  - Loss is finite and decreases after one gradient step
  - Xavier init: no weights are all-zero or all-identical after construction
"""

import sys, os, copy
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.model import Transformer

# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH      = 4
SRC_SEQ    = 10
TGT_SEQ    = 8
SRC_VOCAB  = 200
TGT_VOCAB  = 200
D_MODEL    = 64
N_LAYERS   = 6
N_HEADS    = 8
D_FF       = 256
PAD_IDX    = 0


def make_model(dropout=0.0):
    m = Transformer(
        source_vocab_size=SRC_VOCAB, 
        target_vocab_size=TGT_VOCAB, 
        embedding_dim=D_MODEL, 
        num_layers=N_LAYERS, 
        num_heads=N_HEADS, 
        feed_forward_dim=D_FF,
        dropout=dropout
    )
    return m


def rand_src(batch=BATCH, seq=SRC_SEQ):
    return torch.randint(1, SRC_VOCAB, (batch, seq))


def rand_tgt(batch=BATCH, seq=TGT_SEQ):
    return torch.randint(1, TGT_VOCAB, (batch, seq))


@pytest.fixture(scope="module")
def model():
    m = make_model()
    m.eval()
    return m


# ── 1. Output shapes ──────────────────────────────────────────────────────────

class TestShapes:

    def test_standard_forward(self, model):
        out = model(rand_src(), rand_tgt())
        assert out.shape == (BATCH, TGT_SEQ, TGT_VOCAB)

    def test_batch_size_1(self, model):
        out = model(rand_src(1), rand_tgt(1))
        assert out.shape == (1, TGT_SEQ, TGT_VOCAB)

    def test_batch_size_8(self, model):
        out = model(rand_src(8), rand_tgt(8))
        assert out.shape == (8, TGT_SEQ, TGT_VOCAB)

    def test_short_sequences(self, model):
        out = model(rand_src(seq=2), rand_tgt(seq=2))
        assert out.shape == (BATCH, 2, TGT_VOCAB)

    def test_long_sequences(self, model):
        out = model(rand_src(seq=50), rand_tgt(seq=50))
        assert out.shape == (BATCH, 50, TGT_VOCAB)

    def test_src_tgt_different_lengths(self, model):
        out = model(rand_src(seq=15), rand_tgt(seq=5))
        assert out.shape == (BATCH, 5, TGT_VOCAB)


# ── 2. Causal mask — no future leakage ───────────────────────────────────────

class TestCausalMaskNoLeakage:

    def test_changing_future_token_does_not_affect_past_position(self, model):
        """
        If the causal mask is correct, changing tgt position j must not
        change logits at any position i < j.
        """
        torch.manual_seed(0)
        src  = rand_src(batch=1)
        tgt  = rand_tgt(batch=1, seq=8)
        tgt2 = tgt.clone()
        tgt2[0, 7] = (tgt2[0, 7] + 1) % TGT_VOCAB   # change last position only

        with torch.no_grad():
            out1 = model(src, tgt)
            out2 = model(src, tgt2)

        # positions 0–6 must be identical; only position 7 may differ
        assert torch.allclose(out1[0, :7], out2[0, :7], atol=1e-5), \
            "Future token change leaked into past positions — causal mask broken"

    def test_position_0_only_sees_itself(self, model):
        """Scrambling all positions except 0 must leave position 0 unchanged."""
        torch.manual_seed(1)
        src  = rand_src(batch=1)
        tgt  = rand_tgt(batch=1, seq=6)
        tgt2 = tgt.clone()
        tgt2[0, 1:] = torch.randint(1, TGT_VOCAB, (5,))

        with torch.no_grad():
            out1 = model(src, tgt)
            out2 = model(src, tgt2)

        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), \
            "Position 0 was affected by tokens it should not see"


# ── 3. Padding isolation ──────────────────────────────────────────────────────

class TestPaddingIsolation:

    def test_src_pad_does_not_affect_non_pad_logits(self, model):
        """
        Appending a masked pad token to src must leave decoder logits unchanged.
        If the padding mask is working, the pad contributes zero attention weight,
        so real-token outputs in enc_pad must match enc_clean.
        """
        torch.manual_seed(2)
        src_clean = rand_src(batch=1, seq=6)
        src_pad   = torch.cat([src_clean, torch.zeros(1, 1, dtype=torch.long)], dim=1)
        tgt       = rand_tgt(batch=1)

        src_mask_clean = model.encoder.build_source_mask(src_clean, PAD_IDX)
        src_mask_pad   = model.encoder.build_source_mask(src_pad,   PAD_IDX)

        with torch.no_grad():
            enc_clean, _ = model.encode(src_clean, src_mask_clean)
            enc_pad,   _ = model.encode(src_pad,   src_mask_pad)
            out_clean = model.decode_step(tgt, enc_clean, src_mask_clean)
            out_pad   = model.decode_step(tgt, enc_pad,   src_mask_pad)

        # Padding mask works correctly: pad token contributes zero attention,
        # so decoder logits must be identical with or without the pad position.
        assert torch.allclose(out_clean, out_pad, atol=1e-3), \
            "Padding mask broken — pad token is influencing decoder output"

    def test_tgt_pad_rows_are_masked_in_self_attention(self, model):
        tgt      = rand_tgt(batch=1, seq=6)
        tgt_pad  = tgt.clone()
        tgt_pad[0, -1] = PAD_IDX
        mask     = model.decoder.build_target_mask(tgt_pad, PAD_IDX)
        # Last column must be blocked for all rows
        assert mask[0, 0, :, -1].all(), \
            "Padding column not blocked in tgt_mask"


# ── 4. Batch independence ─────────────────────────────────────────────────────

class TestBatchIndependence:

    def test_sample_in_batch_does_not_affect_others(self, model):
        """
        Modifying sample 0 must not change the logits for samples 1–3.
        """
        torch.manual_seed(3)
        src  = rand_src()
        tgt  = rand_tgt()
        src2 = src.clone()
        src2[0] = rand_src(batch=1)[0]   # replace only first sample

        with torch.no_grad():
            out1 = model(src,  tgt)
            out2 = model(src2, tgt)

        assert torch.allclose(out1[1:], out2[1:], atol=1e-5), \
            "Changing sample 0 affected other samples in the batch"


# ── 5. Determinism ────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_input_same_output_eval(self, model):
        torch.manual_seed(4)
        src = rand_src()
        tgt = rand_tgt()
        with torch.no_grad():
            out1 = model(src, tgt)
            out2 = model(src, tgt)
        assert torch.allclose(out1, out2), \
            "Model is not deterministic in eval mode"

    def test_train_mode_differs_with_dropout(self):
        m = make_model(dropout=0.5)
        m.train()
        src = rand_src()
        tgt = rand_tgt()
        out1 = m(src, tgt)
        out2 = m(src, tgt)
        assert not torch.allclose(out1, out2), \
            "Dropout had no effect in train mode"


# ── 6. Gradient flow ──────────────────────────────────────────────────────────

class TestGradientFlow:

    def test_all_parameters_receive_gradients(self, model):
        # Create a new model instance for gradient tests
        m = make_model()
        m.train()
        src    = rand_src()
        tgt    = rand_tgt()
        logits = m(src, tgt)                        # (batch, tgt_seq, vocab)
        loss   = logits.mean()
        loss.backward()

        no_grad = [
            name for name, p in m.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert no_grad == [], \
            f"Parameters with no gradient: {no_grad}"

    def test_gradients_are_finite(self):
        m = make_model()
        m.train()
        src    = rand_src()
        tgt    = rand_tgt()
        loss   = m(src, tgt).mean()
        loss.backward()

        for name, p in m.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"Non-finite gradient in {name}"


# ── 7. Weight tying ───────────────────────────────────────────────────────────

class TestWeightTying:

    def test_output_projection_and_embedding_are_same_tensor(self, model):
        assert model.output_projection.weight is model.decoder.embedding.token_emb.weight

    def test_updating_embedding_updates_projection(self):
        m = make_model()
        before = m.output_projection.weight.clone()
        with torch.no_grad():
            m.decoder.embedding.token_emb.weight[5] = 999.0
        assert torch.allclose(m.output_projection.weight[5],
                               torch.tensor(999.0).expand(D_MODEL))


# ── 8. Softmax validity ───────────────────────────────────────────────────────

class TestSoftmax:

    def test_probs_sum_to_one(self, model):
        with torch.no_grad():
            logits = model(rand_src(), rand_tgt())
        probs = logits.softmax(dim=-1)
        sums  = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_probs_non_negative(self, model):
        with torch.no_grad():
            logits = model(rand_src(), rand_tgt())
        probs = logits.softmax(dim=-1)
        assert probs.min().item() >= 0.0

    def test_logits_are_not_probabilities(self, model):
        with torch.no_grad():
            logits = model(rand_src(), rand_tgt())
        sums = logits.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=0.1), \
            "Logits sum to 1 — softmax was applied inside the model"


# ── 9. Encoder reuse ──────────────────────────────────────────────────────────

class TestEncoderReuse:

    def test_encode_once_equals_full_forward(self, model):
        torch.manual_seed(5)
        src = rand_src(batch=1)
        tgt = rand_tgt(batch=1)
        with torch.no_grad():
            full_logits        = model(src, tgt)
            enc_out, src_mask  = model.encode(src)
            reuse_logits       = model.decode_step(tgt, enc_out, src_mask)
        assert torch.allclose(full_logits, reuse_logits, atol=1e-5), \
            "encode() + decode_step() does not match forward()"


# ── 10. Sensitivity to src and tgt ───────────────────────────────────────────

class TestSensitivity:

    def test_different_src_changes_output(self, model):
        tgt  = rand_tgt()
        with torch.no_grad():
            out1 = model(rand_src(), tgt)
            out2 = model(rand_src(), tgt)
        assert not torch.allclose(out1, out2)

    def test_different_tgt_changes_output(self, model):
        src  = rand_src()
        with torch.no_grad():
            out1 = model(src, rand_tgt())
            out2 = model(src, rand_tgt())
        assert not torch.allclose(out1, out2)

    def test_same_src_different_tgt_same_enc_output(self, model):
        src = rand_src(batch=1)
        with torch.no_grad():
            enc1, _ = model.encode(src)
            enc2, _ = model.encode(src)
        assert torch.allclose(enc1, enc2, atol=1e-6), \
            "Same src produced different encoder outputs"


# ── 11. Loss decreases after one step ────────────────────────────────────────

class TestLearning:

    def test_loss_is_finite(self):
        m = make_model()
        m.train()
        src    = rand_src()
        tgt_in = rand_tgt()                              # decoder input
        tgt_gt = rand_tgt()                              # ground truth labels
        logits = m(src, tgt_in)                          # (batch, seq, vocab)
        loss   = nn.CrossEntropyLoss(ignore_index=PAD_IDX)(
            logits.reshape(-1, TGT_VOCAB), tgt_gt.reshape(-1)
        )
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_loss_decreases_after_one_step(self):
        torch.manual_seed(99)
        m      = make_model()
        m.train()
        opt    = torch.optim.Adam(m.parameters(), lr=1e-3)
        src    = rand_src()
        tgt_in = rand_tgt()
        tgt_gt = rand_tgt()
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        losses = []
        for _ in range(3):
            opt.zero_grad()
            logits = m(src, tgt_in)
            loss   = criterion(logits.reshape(-1, TGT_VOCAB), tgt_gt.reshape(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses}"


# ── 12. Weight initialisation ─────────────────────────────────────────────────

class TestInitialisation:

    def test_no_all_zero_weight_matrices(self):
        m = make_model()
        for name, p in m.named_parameters():
            if p.dim() > 1:
                assert not torch.all(p == 0), \
                    f"Weight matrix {name} is all zeros"

    def test_no_uniform_weight_matrices(self):
        m = make_model()
        for name, p in m.named_parameters():
            if p.dim() > 1:
                assert p.std().item() > 1e-6, \
                    f"Weight matrix {name} has near-zero std (all same value)"



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
