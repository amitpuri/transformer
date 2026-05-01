# 🤖 Transformer Architecture 

A clean, modular, and highly documented implementation of the original Transformer architecture from the paper **"Attention Is All You Need"** (Vaswani et al., 2017).

This project is designed for **readability** and **educational purposes**, breaking down the complex architecture into understandable building blocks with an interactive CLI to explore each component step by step.

## 📁 Project Structure

```
transformer/
├── transformer/              # Core library
│   ├── model.py              # Full Transformer assembly
│   ├── encoder.py            # Encoder stack and repeating layers
│   ├── decoder.py            # Decoder stack with masked and cross-attention
│   ├── layers.py             # Multi-Head Attention, Feed-Forward, LayerNorm
│   ├── embeddings.py         # Token embeddings + Sinusoidal Positional Encoding
│   └── tokenization.py       # Word-level (spaCy) + Subword (BPE) tokenizers
├── main.py                   # Interactive CLI menu
├── trace_pipeline.py         # Step-by-step text → embedding visualization
├── generation_demo.py        # Autoregressive text generation demo
├── tests/                    # Test suite
└── requirements.txt
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Language Models
SpaCy requires downloading language models separately:
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### 3. Run the Interactive Menu
Explore the model through a user-friendly CLI:
```bash
python main.py
```

This opens a Rich-powered menu with three exploration modes:

| Option | Description |
|--------|-------------|
| **1. Trace Embedding Pipeline** | Traces raw text -> tokens -> IDs -> embeddings step by step |
| **2. View Model Architecture** | Instantiates a sample Transformer and prints its parameter summary |
| **3. Text Generation Demo** | Runs autoregressive decoding with sampling (untrained weights) |

### Deep Trace Mode

Toggle option 4 to enable **Deep Trace Mode**. When enabled, the model will output every step of its execution, including tensor shapes, scaling logic, and intermediate results.

For example, tracing the Embedding Pipeline with Deep Trace ON shows:

```
+- Tokenizer.tokenize  lang='en'
|  Input text: "I like to learn about AI."
|  Using spaCy rule-based tokenizer
|  Tokens (7): ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
+-  7 tokens
Step 1: Tokenization -> ['I', 'like', 'to', 'learn', 'about', 'AI', '.']

+- Tokenizer.build_vocab  lang='en', min_freq=1, 1 sentences
|  +- Tokenizer.tokenize  lang='en'
|  |  Tokens (7): ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
|  +-  7 tokens
|  Vocabulary size: 11 (4 special + 7 learned tokens)
|  Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
+-  vocab_size=11

+- Tokenizer.encode  add_special=True
|  Token->ID mapping: [('I', 6), ('like', 9), ('to', 10), ('learn', 8), ('about', 7), ('AI', 5), ('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 6, 9, 10, 8, 7, 5, 4, 3]
+-  9 token IDs

+- TransformerEmbedding  vocab_size=11, dim=64
|  [*] Input token IDs: shape=(1, 9)
|  TokenEmbedding: lookup table -> dense vectors, * sqrt(64) = * 8.00 scaling
|  [*] Token embeddings (scaled): shape=(1, 9, 64)
|  PositionalEncoding: sin/cos signals for positions 0..8
|  [*] Positional encoding: shape=(9, 64)
|  Combined: token_embeddings + positional_encoding -> dropout
|  [*] Output embeddings: shape=(1, 9, 64)
+-  shape (1, 9, 64)
```

And tracing the Generation Decoder step shows how cross-attention integrates encoder context:

```
-- Generation step 1 (Current input: <bos>) --
+- Transformer.decode_step  target_len=1
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 1)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 1, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..0
|  |  |  [*] Positional encoding: shape=(1, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 1, 128)
|  |  +-  shape (1, 1, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 1, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 1)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 1, 32)
|  |  |  |  +-  shape (1, 4, 1, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 1, 32)
|  |  |  |  +-  shape (1, 4, 1, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 1, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 1, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 1)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 1, 32)
|  |  |  |  +-  shape (1, 4, 1, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 1, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 1, 32)
|  |  |  |  +-  shape (1, 4, 1, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 1, 128)
|  |  |  +-  shape (1, 1, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 1, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 1, 128)
|  |  [*] Decoder output: shape=(1, 1, 128)
|  +-  shape (1, 1, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 1, 56)
Sampled token ID 23 -> 'robot'
```

---

## 🧠 Component Walkthrough

### End-to-End Data Flow

```
"I like to learn about AI."
          │
          ▼
  ┌─── Tokenize ───┐
  │ spaCy / regex   │
  └────────┬────────┘
           ▼
  ['I','like','to','learn','about','AI','.']
           │
           ▼
  ┌── Vocabulary ──┐
  │ token → int ID  │
  └────────┬───────┘
           ▼
  [2, 6, 9, 10, 8, 7, 5, 4, 3]  (with <bos>=2, <eos>=3)
           │
           ▼
  ┌── Token Embedding ──┐
  │ ID → 64-dim vector   │
  │ × √d scaling         │
  └────────┬─────────────┘
           │
           + ←── Positional Encoding (sin/cos signals)
           │
           ▼
  ┌──── Encoder (×N layers) ────┐
  │  Self-Attention + Add&Norm  │
  │  FeedForward   + Add&Norm   │
  └────────────┬────────────────┘
               │ (contextual representations)
               ▼
  ┌──── Decoder (×N layers) ────────────┐
  │  Masked Self-Attention + Add&Norm   │
  │  Cross-Attention       + Add&Norm   │  ← reads encoder output
  │  FeedForward           + Add&Norm   │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  ┌── Output Projection ──┐
  │  Linear(d → vocab)    │  ← weight-tied with decoder embeddings
  └────────────┬──────────┘
               │
               ▼
         logits → softmax → next token prediction
```

---

### 1. Tokenization — `tokenization.py`

#### SpacyTokenizer (Word-level)

Uses spaCy's `en_core_web_sm` model for rule-based tokenization that handles punctuation splitting, contractions, etc. Falls back to a simple regex `\w+|[^\w\s]` if spaCy models are missing.

```
Input:  "I like to learn about AI."
Output: ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
```

#### Vocabulary

Manages the bidirectional mapping between string tokens and integer IDs:
- Counts token frequencies with `collections.Counter`
- Reserves 4 special tokens first: `<pad>=0, <unk>=1, <bos>=2, <eos>=3`
- Only keeps tokens appearing ≥ `min_freq` times
- `encode()` maps tokens → IDs and wraps with `<bos>` / `<eos>`
- `decode()` maps IDs back → tokens, optionally skipping special tokens

#### SubwordTokenizer (BPE)

A full **Byte Pair Encoding** implementation (the same algorithm used by GPT models):
1. Splits all words to characters with an end-of-word marker `</w>`
2. Iteratively finds the most frequent adjacent character pair and merges them
3. Stops when `vocab_size` is reached
4. Handles rare words gracefully by breaking them into known subword pieces

---

### 2. Embeddings — `embeddings.py`

#### TokenEmbedding

A standard `nn.Embedding` lookup table: each token in the vocabulary gets a learnable dense vector. `padding_idx=0` ensures the `<pad>` token always stays as a zero vector.

#### SinusoidalPositionalEncoding

Transformers process tokens in parallel (no recurrence), so they need this signal to understand word order.

```python
pe[:, 0::2] = sin(position / 10000^(2i/d))   # even dimensions
pe[:, 1::2] = cos(position / 10000^(2i/d))    # odd dimensions
```

- Each position gets a unique "fingerprint" vector; nearby positions have similar signals
- Registered as a **buffer** — not a learnable parameter; these are fixed signals

#### TransformerEmbedding

Combines both sub-components:

```python
tokens = token_emb(x) * √embedding_dim   # scale up embeddings
positions = pos_enc(x)                     # get positional signals
return dropout(tokens + positions)         # add + regularize
```

- **Scaling by √d**: Makes embedding magnitudes comparable to the positional encoding magnitudes (standard practice from the paper)

---

### 3. Core Layers — `layers.py`

#### Scaled Dot-Product Attention

The **core mechanism** of the entire Transformer. Every token "asks a question" (Query), "advertises what it contains" (Key), and "shares its information" (Value).

```
scores  = Q × Kᵀ / √d_k        ← similarity scores
weights = softmax(scores)       ← attention probabilities  
output  = weights × V           ← weighted combination of values
```

- **Scaling by √d_k** prevents dot products from growing too large (which would push softmax into regions with near-zero gradients)
- **Masking**: Positions set to `-inf` become 0 after softmax — used for padding tokens and future-token masking

#### Multi-Head Attention

Instead of one big attention operation, the model runs **multiple smaller ones in parallel**, each learning different relationships:

```
embedding_dim=128, num_heads=4  →  head_dim = 128/4 = 32
```

1. **Linear projections**: `Q, K, V = W_q(x), W_k(x), W_v(x)`
2. **Split into heads**: Reshape `(batch, seq, 128)` → `(batch, 4, seq, 32)`
3. **Parallel attention**: Run ScaledDotProductAttention on all heads simultaneously
4. **Merge heads**: Reshape back to `(batch, seq, 128)`
5. **Output projection**: Final linear layer mixes information across heads

> **Why multiple heads?** Each head can learn different relationships — Head 1 might learn syntactic patterns (subject-verb), Head 2 might learn semantic similarity, Head 3 might learn positional patterns. Multiple heads = richer representations.

#### Position-wise Feed-Forward Network

```python
nn.Sequential(
    nn.Linear(128, 512),   # expand (4× embedding dim)
    nn.ReLU(),             # non-linearity
    nn.Dropout(0.1),       # regularization
    nn.Linear(512, 128)    # compress back
)
```

- Applied **identically** to each token position (no cross-token interaction)
- The attention layer decides *what to look at*; the FFN decides *what to do with it*

#### Layer Normalization

```python
output = alpha * (x - mean) / (std + epsilon) + bias
```

- Normalizes each token's feature vector to zero mean and unit variance
- `alpha` (scale) and `bias` (shift) are **learnable**
- Stabilizes training by keeping activations in a well-behaved range

---

### 4. Encoder — `encoder.py`

#### EncoderLayer

Each encoder layer follows this pattern:

```
x ──→ Self-Attention ──→ Add & Norm ──→ FeedForward ──→ Add & Norm ──→ output
  │                  ↑                │              ↑
  └──── residual ────┘                └── residual ──┘
```

- **Self-Attention**: `Q = K = V = x` — every token attends to every other token in the sequence
- **Residual connections** (`x + sublayer(x)`) let gradients flow directly through the network, enabling deep stacking

#### TransformerEncoder

Stacks `N` EncoderLayers on top of a `TransformerEmbedding`. Builds a **source mask** to ignore `<pad>` tokens.

---

### 5. Decoder — `decoder.py`

#### DecoderLayer

Each decoder layer has **three** sub-layers (vs encoder's two):

```
x ──→ Masked Self-Attn ──→ Add&Norm ──→ Cross-Attn ──→ Add&Norm ──→ FFN ──→ Add&Norm
       (Q=K=V=x)                         (Q=x, K=V=encoder_output)
```

1. **Masked Self-Attention**: Same as encoder, but with a **causal mask** preventing token `i` from seeing tokens `i+1, i+2, ...` — ensures the model can only use past context
2. **Cross-Attention**: Query from decoder, Key+Value from encoder — this is how the decoder "reads" the source sentence
3. **Feed-Forward**: Same per-position transformation as the encoder

#### Target Mask (Look-Ahead Mask)

Combines padding mask + upper-triangular causal mask:

```
         tok1  tok2  tok3  tok4
tok1  [  ✓     ✗     ✗     ✗  ]   ← can only see itself
tok2  [  ✓     ✓     ✗     ✗  ]   ← can see tok1, tok2
tok3  [  ✓     ✓     ✓     ✗  ]   ← can see tok1-3
tok4  [  ✓     ✓     ✓     ✓  ]   ← can see everything
```

#### TransformerDecoder

Stacks `N` DecoderLayers on top of a `TransformerEmbedding`.

---

### 6. Full Model — `model.py`

```python
class Transformer(nn.Module):
    encoder           = TransformerEncoder(...)
    decoder           = TransformerDecoder(...)
    output_projection = nn.Linear(embedding_dim, target_vocab_size)
```

**Key design decisions**:

- **Weight Tying**: The decoder's token embedding weights are **shared** with the output projection layer — reduces parameters and improves generalization
- **Xavier Uniform Init**: All parameters with `dim > 1` are initialized with `xavier_uniform_` for stable training
- **Inference helpers**: `encode()` runs the encoder once; `decode_step()` runs a single autoregressive decoding step

#### Sample Architecture

| Component | Details |
|-----------|---------|
| Embedding Dim | 128 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Attention Heads | 4 |
| Feed Forward Dim | 512 |
| Total Parameters | 1,182,696 |

---

### 7. Text Generation — `generation_demo.py`

Demonstrates autoregressive decoding with **temperature-scaled sampling**:

```python
for i in range(max_len):
    logits = model.decode_step(tgt_tensor, enc_out, src_mask)
    next_token_logits = logits[0, -1, :] / temperature
    probs = torch.softmax(next_token_logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    generated_ids.append(next_id)
```

| Concept | Explanation |
|---------|-------------|
| **Autoregressive** | Each generated token is fed back as input for the next step |
| **Temperature** | Controls randomness: higher = more diverse; lower = more deterministic |
| **`torch.multinomial`** | Samples from the probability distribution (not greedy argmax) |

> **Note**: Output is random because the model has untrained weights. With proper training on a parallel corpus, this same architecture produces coherent translations.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 🧠 Key Features

- **In-depth Documentation**: Every class and method includes docstrings explaining the *why* behind the *what*, referencing the original paper
- **Visual Trace**: See exactly how a string of text is transformed into numerical vectors ready for the attention mechanism
- **Interactive CLI**: Explore each component through a Rich-powered terminal interface
- **Modular Design**: Each component is self-contained and reusable

## 📖 Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need.** *Advances in neural information processing systems*, 30.
