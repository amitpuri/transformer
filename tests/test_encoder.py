import sys, os
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.encoder import TransformerEncoder

BATCH      = 2
SEQ        = 10
VOCAB      = 100
D_MODEL    = 64
N_LAYERS   = 6
N_HEADS    = 8
D_FF       = 256
PAD_IDX    = 0


@pytest.fixture(scope="module")
def encoder():
    return TransformerEncoder(
        vocab_size=VOCAB, 
        embedding_dim=D_MODEL, 
        num_layers=N_LAYERS, 
        num_heads=N_HEADS, 
        feed_forward_dim=D_FF
    )


class TestEncoder:

    def test_output_shape(self, encoder):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        out = encoder(src)
        assert out.shape == (BATCH, SEQ, D_MODEL)

    def test_num_layers(self, encoder):
        assert len(encoder.layers) == N_LAYERS

    def test_padding_mask_shape(self, encoder):
        src  = torch.randint(1, VOCAB, (BATCH, SEQ))
        src[0, -2:] = PAD_IDX
        mask = encoder.build_source_mask(src, PAD_IDX)
        assert mask.shape == (BATCH, 1, 1, SEQ)

    def test_padding_positions_masked(self, encoder):
        src = torch.randint(1, VOCAB, (BATCH, SEQ))
        src[0, -2:] = PAD_IDX
        mask = encoder.build_source_mask(src, PAD_IDX)
        assert mask[0, 0, 0, -2].item() == True
        assert mask[0, 0, 0,  0].item() == False

    def test_pad_tokens_dont_affect_other_positions(self, encoder):
        encoder.eval()
        # Test: Real tokens at start must produce identical outputs 
        # regardless of how many padding tokens follow them, as long as they are masked.
        real_tokens = torch.randint(1, VOCAB, (1, 5))
        src1 = torch.cat([real_tokens, torch.tensor([[PAD_IDX, PAD_IDX]], dtype=torch.long)], dim=1)
        src2 = torch.cat([real_tokens, torch.tensor([[PAD_IDX]], dtype=torch.long)], dim=1)

        mask1 = encoder.build_source_mask(src1, PAD_IDX)
        mask2 = encoder.build_source_mask(src2, PAD_IDX)
        
        with torch.no_grad():
            out1 = encoder(src1, mask1)
            out2 = encoder(src2, mask2)

        # non-padding positions must be unaffected by the number of trailing pad tokens
        assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-4)


    def test_output_differs_across_layers(self, encoder):
        encoder.eval()
        src = torch.randint(1, VOCAB, (1, SEQ))
        with torch.no_grad():
            after_emb = encoder.embedding(src)
            after_l1  = encoder.layers[0](after_emb)
            after_l6  = encoder(src)
        assert not torch.allclose(after_emb, after_l1)
        assert not torch.allclose(after_l1,  after_l6)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
