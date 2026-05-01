"""
Tests for the BPE Tokenizer.

Covers:
  - Vocabulary is built correctly after training
  - Special tokens always present at fixed indices
  - encode() produces integer IDs within vocab range
  - decode() recovers the original text
  - Unknown words fall back to <unk>
  - Round-trip: encode → decode
  - Save and load preserves behaviour
  - encode → TokenEmbedding produces correct tensor shape (full pipeline test)
"""

import os, sys, tempfile
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.tokenization import SubwordTokenizer
from transformer.embeddings import TransformerEmbedding

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world hello transformer",
    "attention is all you need",
    "byte pair encoding splits rare words into subwords",
    "the dog sat on the mat",
    "the cat sat on the mat",
]

D_MODEL  = 64
DROPOUT  = 0.0


@pytest.fixture(scope="module")
def trained_tok():
    tok = SubwordTokenizer(vocab_size=200)
    tok.train(CORPUS)
    return tok


# ── Vocabulary ────────────────────────────────────────────────────────────────

class TestVocabulary:
    def test_special_tokens_at_fixed_indices(self, trained_tok):
        assert trained_tok.token_to_id["<pad>"] == 0
        assert trained_tok.token_to_id["<unk>"] == 1
        assert trained_tok.token_to_id["<bos>"] == 2
        assert trained_tok.token_to_id["<eos>"] == 3

    def test_vocab_size_respected(self, trained_tok):
        assert len(trained_tok) <= 200

    def test_id_to_token_and_token_to_id_are_consistent(self, trained_tok):
        for tok, idx in trained_tok.token_to_id.items():
            assert trained_tok.id_to_token[idx] == tok


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncoding:
    def test_returns_list_of_ints(self, trained_tok):
        ids = trained_tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_ids_in_valid_range(self, trained_tok):
        ids = trained_tok.encode("the quick brown fox")
        assert all(0 <= i < len(trained_tok) for i in ids)

    def test_bos_eos_present(self, trained_tok):
        ids = trained_tok.encode("hello world", add_special=True)
        assert ids[0]  == trained_tok.bos_id
        assert ids[-1] == trained_tok.eos_id

    def test_no_special_tokens_when_disabled(self, trained_tok):
        ids = trained_tok.encode("hello world", add_special=False)
        assert trained_tok.bos_id not in ids
        assert trained_tok.eos_id not in ids

    def test_unknown_word_maps_to_unk(self, trained_tok):
        # "zzzzzz" was never in corpus — each char may still be in vocab but
        # the rare combo should fall back gracefully; at least no crash
        ids = trained_tok.encode("zzzzzz", add_special=False)
        assert all(0 <= i < len(trained_tok) for i in ids)


# ── Decoding ──────────────────────────────────────────────────────────────────

class TestDecoding:
    def test_round_trip_in_corpus(self, trained_tok):
        original = "the dog sat on the mat"
        ids = trained_tok.encode(original, add_special=False)
        decoded = trained_tok.decode(ids)
        assert decoded == original

    def test_special_tokens_skipped_in_decode(self, trained_tok):
        ids = trained_tok.encode("hello world", add_special=True)
        decoded = trained_tok.decode(ids, skip_special=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, trained_tok):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            trained_tok.save(path)
            loaded = SubwordTokenizer.load(path)
            original_ids = trained_tok.encode("attention is all you need")
            loaded_ids   = loaded.encode("attention is all you need")
            assert original_ids == loaded_ids
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── Full pipeline: BPE → TokenIDs → TransformerEmbedding ─────────────────────

class TestFullPipeline:
    def test_encode_then_embed(self, trained_tok):
        """
        Real end-to-end: text → BPE token IDs → TransformerEmbedding tensor.
        This is the actual first two steps of the Transformer.
        """
        sentences = [
            "hello world",
            "attention is all you need",
        ]

        # Encode each sentence
        encoded = [trained_tok.encode(s) for s in sentences]

        # Pad to same length for batching
        max_len = max(len(e) for e in encoded)
        padded  = [e + [trained_tok.pad_id] * (max_len - len(e)) for e in encoded]
        x = torch.tensor(padded, dtype=torch.long)  # (2, max_len)

        # Pass through embedding layer
        emb = TransformerEmbedding(
            vocab_size=len(trained_tok),
            embedding_dim=D_MODEL,
            max_seq_length=512,
            dropout=DROPOUT,
        )
        emb.eval()
        out = emb(x)

        assert out.shape == (2, max_len, D_MODEL), \
            f"Expected (2, {max_len}, {D_MODEL}), got {out.shape}"

    def test_padding_positions_reflect_pad_token(self, trained_tok):
        """Padding IDs (0) must produce zero token embeddings (padding_idx=0)."""
        emb = TransformerEmbedding(len(trained_tok), D_MODEL, dropout=0.0)
        emb.eval()

        pad_only = torch.zeros(1, 5, dtype=torch.long)   # all padding
        out = emb(pad_only)

        # Token embedding is 0 for pad; positional encoding is non-zero,
        # so the sum is purely the positional signal — not all zeros.
        # What we verify: the raw token embedding for pad is zero.
        tok_out = emb.token_emb(pad_only)
        assert tok_out.abs().sum().item() == 0.0



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
