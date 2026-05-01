"""
Tests for the spaCy Tokenizer + Vocabulary pipeline.
"""

import os, sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer.tokenization import SpacyTokenizer, Vocabulary
from transformer.embeddings import TransformerEmbedding

EN_CORPUS = [
    "attention is all you need",
    "the transformer model uses self attention",
    "I love building neural networks",
    "the quick brown fox jumps over the lazy dog",
    "transformers have replaced recurrent models",
]

DE_CORPUS = [
    "Aufmerksamkeit ist alles was du brauchst",
    "das Transformer Modell nutzt Selbstaufmerksamkeit",
    "ich liebe neuronale Netzwerke",
]

D_MODEL = 64


@pytest.fixture(scope="module")
def tokenizer():
    tok = SpacyTokenizer()
    tok.build_vocab(EN_CORPUS, lang="en", min_freq=1)
    tok.build_vocab(DE_CORPUS, lang="de", min_freq=1)
    return tok


# ── Tokenization (text → string tokens) ──────────────────────────────────────

class TestTokenization:
    def test_tokenize_splits_into_words(self, tokenizer):
        tokens = tokenizer.tokenize("attention is all you need", lang="en")
        assert tokens == ["attention", "is", "all", "you", "need"]

    def test_de_splits_into_words(self, tokenizer):
        tokens = tokenizer.tokenize("ich liebe neuronale Netzwerke", lang="de")
        assert tokens == ["ich", "liebe", "neuronale", "Netzwerke"]

    def test_punctuation_is_a_separate_token(self, tokenizer):
        tokens = tokenizer.tokenize("Hello, world!", lang="en")
        assert "," in tokens
        assert "!" in tokens


# ── Vocabulary ────────────────────────────────────────────────────────────────

class TestVocabulary:
    def test_special_tokens_at_fixed_indices(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        assert vocab_en.pad_id == 0
        assert vocab_en.unk_id == 1
        assert vocab_en.bos_id == 2
        assert vocab_en.eos_id == 3

    def test_corpus_words_in_vocab(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        assert "attention" in vocab_en.token_to_id
        assert "transformer" in vocab_en.token_to_id

    def test_min_freq_filters_rare_tokens(self):
        tok = SpacyTokenizer()
        # "rare" appears once, "the" appears twice
        tok.build_vocab(["the cat the dog", "rare word here"], lang="en", min_freq=2)
        vocab_en = tok.vocabs["en"]
        assert "rare" not in vocab_en.token_to_id
        assert "the" in vocab_en.token_to_id

    def test_unknown_token_maps_to_unk(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        assert vocab_en["nonexistentxyz"] == vocab_en.unk_id


# ── Encoding ──────────────────────────────────────────────────────────────────

class TestEncoding:
    def test_returns_list_of_ints(self, tokenizer):
        ids = tokenizer.encode("attention is all you need", lang="en")
        assert all(isinstance(i, int) for i in ids)

    def test_bos_eos_wrapping(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        ids = tokenizer.encode("attention is all", lang="en", add_special=True)
        assert ids[0] == vocab_en.bos_id
        assert ids[-1] == vocab_en.eos_id

    def test_no_special_tokens(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        ids = tokenizer.encode("attention is all", lang="en", add_special=False)
        assert vocab_en.bos_id not in ids
        assert vocab_en.eos_id not in ids

    def test_ids_in_valid_range(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        ids = tokenizer.encode("attention is all you need", lang="en")
        assert all(0 <= i < len(vocab_en) for i in ids)


# ── Decoding ──────────────────────────────────────────────────────────────────

class TestDecoding:
    def test_round_trip(self, tokenizer):
        sentence = "attention is all you need"
        ids = tokenizer.encode(sentence, lang="en", add_special=False)
        decoded = tokenizer.decode(ids, lang="en")
        assert decoded == sentence

    def test_special_tokens_skipped(self, tokenizer):
        ids = tokenizer.encode("attention is all", lang="en")
        decoded = tokenizer.decode(ids, lang="en", skip_special=True)
        assert "<bos>" not in decoded and "<eos>" not in decoded


# ── Full pipeline: text → spaCy → IDs → TransformerEmbedding ─────────────────

class TestFullPipeline:
    def test_shape(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        sentences = ["attention is all you need", "the transformer model"]
        encoded = [tokenizer.encode(s, lang="en") for s in sentences]

        max_len = max(len(e) for e in encoded)
        padded = [e + [vocab_en.pad_id] * (max_len - len(e)) for e in encoded]
        x = torch.tensor(padded, dtype=torch.long)   # (2, max_len)

        emb = TransformerEmbedding(
            vocab_size=len(vocab_en),
            embedding_dim=D_MODEL,
            dropout=0.0,
        )
        emb.eval()
        out = emb(x)
        assert out.shape == (2, max_len, D_MODEL)

    def test_different_sentences_produce_different_embeddings(self, tokenizer):
        vocab_en = tokenizer.vocabs["en"]
        emb = TransformerEmbedding(len(vocab_en), D_MODEL, dropout=0.0)
        emb.eval()

        ids_a = tokenizer.encode("attention is all you need", lang="en")
        ids_b = tokenizer.encode("the transformer model uses self attention", lang="en")

        xa = torch.tensor([ids_a], dtype=torch.long)
        xb = torch.tensor([ids_b], dtype=torch.long)

        out_a = emb(xa)
        out_b = emb(xb)
        # Position 0 is <bos> for both — identical by design.
        # Position 1 is the first real word token and must differ.
        assert not torch.allclose(out_a[0, 1], out_b[0, 1])



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
