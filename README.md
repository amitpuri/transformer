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

```
$ python main.py
                                         ╭────────────────────────── Welcome ───────────────────────────╮
                                         │ Transformer Exploration Toolkit                              │
                                         │ An educational implementation of 'Attention Is All You Need' │
                                         ╰──────────────────────────────────────────────────────────────╯

Main Menu
1. Trace Embedding Pipeline (Step-by-step visualization)
2. View Model Architecture (Summary table)
3. Text Generation Demo (Greedy decoding process)
4. Toggle Deep Trace Mode (Currently: OFF)
5. Exit
Select an option [1/2/3/4/5]: 4
Deep trace mode is now ON.

Main Menu
1. Trace Embedding Pipeline (Step-by-step visualization)
2. View Model Architecture (Summary table)
3. Text Generation Demo (Greedy decoding process)
4. Toggle Deep Trace Mode (Currently: ON)
5. Exit
Select an option [1/2/3/4/5]: 1

Tracing Embedding Pipeline for: 'I like to learn about AI.'

+- Tokenizer.tokenize  lang='en'
|  Input text: "I like to learn about AI."
|  Using spaCy rule-based tokenizer
|  Tokens (7): ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
+-  7 tokens
Step 1: Tokenization -> ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
+- Tokenizer.build_vocab  lang='en', min_freq=1, 1 sentences
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "I like to learn about AI."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
|  +-  7 tokens
|  Vocabulary size: 11 (4 special + 7 learned tokens)
|  Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
+-  vocab_size=11
+- Tokenizer.encode  add_special=True
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "I like to learn about AI."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['I', 'like', 'to', 'learn', 'about', 'AI', '.']
|  +-  7 tokens
|  Token->ID mapping: [('I', 6), ('like', 9), ('to', 10), ('learn', 8), ('about', 7), ('AI', 5), ('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 6, 9, 10, 8, 7, 5, 4, 3]
+-  9 token IDs
   Step 2:
  Vocabulary
   Mapping
┏━━━━━━━┳━━━━┓
┃ Token ┃ ID ┃
┡━━━━━━━╇━━━━┩
│ <bos> │ 2  │
│ I     │ 6  │
│ like  │ 9  │
│ to    │ 10 │
│ learn │ 8  │
│ about │ 7  │
│ AI    │ 5  │
│ .     │ 4  │
│ <eos> │ 3  │
└───────┴────┘
+- TransformerEmbedding  vocab_size=11, dim=64
|  [*] Input token IDs: shape=(1, 9)
|  TokenEmbedding: lookup table -> dense vectors, * sqrt(64) = * 8.00 scaling
|  [*] Token embeddings (scaled): shape=(1, 9, 64)
|  PositionalEncoding: sin/cos signals for positions 0..8
|  [*] Positional encoding: shape=(9, 64)
|  Combined: token_embeddings + positional_encoding -> dropout
|  [*] Output embeddings: shape=(1, 9, 64)
+-  shape (1, 9, 64)

Step 3: Multi-dimensional Representation
Final shape: (1, 9, 64) (Batch, Seq, Dim)
   Final Embedding Samples (First 4 dims)
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Token ┃ Embedding Vector                 ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ <bos> │ [4.23, 4.088, 6.751, 0.614]      │
│ I     │ [14.6, 8.786, -13.264, -10.418]  │
│ like  │ [-10.036, -7.936, -4.459, 0.925] │
│ to    │ [9.131, 2.131, -0.93, 8.575]     │
│ learn │ [-9.781, -12.986, 1.084, 8.534]  │
│ about │ [-3.853, 24.921, 2.404, -1.624]  │
│ AI    │ [8.001, -2.981, 2.073, 0.633]    │
│ .     │ [14.565, -9.035, -9.432, 5.16]   │
│ <eos> │ [3.711, 9.546, -11.296, 7.299]   │
└───────┴──────────────────────────────────┘

Main Menu
1. Trace Embedding Pipeline (Step-by-step visualization)
2. View Model Architecture (Summary table)
3. Text Generation Demo (Greedy decoding process)
4. Toggle Deep Trace Mode (Currently: ON)
5. Exit
Select an option [1/2/3/4/5]: 2
    Transformer Architecture
            (Sample)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Component        ┃ Details   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Embedding Dim    │ 128       │
│ Encoder Layers   │ 2         │
│ Decoder Layers   │ 2         │
│ Attention Heads  │ 4         │
│ Feed Forward Dim │ 512       │
│ Total Parameters │ 1,182,696 │
└──────────────────┴───────────┘

Note: This is a scaled-down version for demonstration.

Main Menu
1. Trace Embedding Pipeline (Step-by-step visualization)
2. View Model Architecture (Summary table)
3. Text Generation Demo (Greedy decoding process)
4. Toggle Deep Trace Mode (Currently: ON)
5. Exit
Select an option [1/2/3/4/5]: 3

Text Generation Demo
Note: The model is untrained (random weights).
Using sampling to visualize the decoding process.

+- Tokenizer.build_vocab  lang='en', min_freq=1, 5 sentences
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "The quick brown fox jumps over the lazy dog."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (10): ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
|  +-  10 tokens
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "Artificial intelligence is transforming the world."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['Artificial', 'intelligence', 'is', 'transforming', 'the', 'world', '.']
|  +-  7 tokens
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "I love learning about deep learning and transformers."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (9): ['I', 'love', 'learning', 'about', 'deep', 'learning', 'and', 'transformers', '.']
|  +-  9 tokens
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "Translate this sentence to another language."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['Translate', 'this', 'sentence', 'to', 'another', 'language', '.']
|  +-  7 tokens
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "network neuron weight bias activation function data science computer vision speech text translation robot future digital spark
logic dream code run fast deep neural system"
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (25): ['network', 'neuron', 'weight', 'bias', 'activation', 'function', 'data', 'science', 'computer', 'vision', 'speech', 'text',
'translation', 'robot', 'future', 'digital', 'spark', 'logic', 'dream', 'code', 'run', 'fast', 'deep', 'neural', 'system']
|  +-  25 tokens
|  Vocabulary size: 56 (4 special + 52 learned tokens)
|  Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
+-  vocab_size=56
Input : The quick brown fox jumps over the lazy dog.
Output: -- Encoding source prompt: 'The quick brown fox jumps over the lazy dog.' --
+- Tokenizer.encode  add_special=True
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "The quick brown fox jumps over the lazy dog."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (10): ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
|  +-  10 tokens
|  Token->ID mapping: [('The', 7), ('quick', 38), ('brown', 14), ('fox', 23), ('jumps', 28), ('over', 37), ('the', 47), ('lazy', 30), ('dog', 20),
('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 7, 38, 14, 23, 28, 37, 47, 30, 20, 4, 3]
+-  12 token IDs
+- Transformer.encode  encode source sequence only
|  Source mask: shape (1, 1, 1, 12)
|  +- TransformerEncoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 12)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 12, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..11
|  |  |  [*] Positional encoding: shape=(12, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 12, 128)
|  |  +-  shape (1, 12, 128)
|  |  -- Encoder Layer 1/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  -- Encoder Layer 2/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  [*] Encoder output: shape=(1, 12, 128)
|  +-  shape (1, 12, 128)
+-  encoder output shape (1, 12, 128)
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
Sampled token ID 26 -> 'intelligence'
intelligence -- Generation step 2 (Current input: <bos> intelligence) --
+- Transformer.decode_step  target_len=2
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 2)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 2, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..1
|  |  |  [*] Positional encoding: shape=(2, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 2, 128)
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  [*] Decoder output: shape=(1, 2, 128)
|  +-  shape (1, 2, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 2, 56)
Sampled token ID 52 -> 'translation'
translation -- Generation step 3 (Current input: <bos> intelligence translation) --
+- Transformer.decode_step  target_len=3
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 3)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 3, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..2
|  |  |  [*] Positional encoding: shape=(3, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 3, 128)
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  [*] Decoder output: shape=(1, 3, 128)
|  +-  shape (1, 3, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 3, 56)
Sampled token ID 52 -> 'translation'
translation -- Generation step 4 (Current input: <bos> intelligence translation translation) --
+- Transformer.decode_step  target_len=4
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 4)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 4, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..3
|  |  |  [*] Positional encoding: shape=(4, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 4, 128)
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  [*] Decoder output: shape=(1, 4, 128)
|  +-  shape (1, 4, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 4, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 5 (Current input: <bos> intelligence translation translation data) --
+- Transformer.decode_step  target_len=5
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 5)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 5, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..4
|  |  |  [*] Positional encoding: shape=(5, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 5, 128)
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  [*] Decoder output: shape=(1, 5, 128)
|  +-  shape (1, 5, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 5, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 6 (Current input: <bos> intelligence translation translation data data) --
+- Transformer.decode_step  target_len=6
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 6)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 6, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..5
|  |  |  [*] Positional encoding: shape=(6, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 6, 128)
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  [*] Decoder output: shape=(1, 6, 128)
|  +-  shape (1, 6, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 6, 56)
Sampled token ID 1 -> '<unk>'
<unk> -- Generation step 7 (Current input: <bos> intelligence translation translation data data <unk>) --
+- Transformer.decode_step  target_len=7
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 7)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 7, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..6
|  |  |  [*] Positional encoding: shape=(7, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 7, 128)
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  [*] Decoder output: shape=(1, 7, 128)
|  +-  shape (1, 7, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 7, 56)
Sampled token ID 1 -> '<unk>'
<unk> -- Generation step 8 (Current input: <bos> intelligence translation translation data data <unk> <unk>) --
+- Transformer.decode_step  target_len=8
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 8)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 8, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..7
|  |  |  [*] Positional encoding: shape=(8, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 8, 128)
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  [*] Decoder output: shape=(1, 8, 128)
|  +-  shape (1, 8, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 8, 56)
Sampled token ID 30 -> 'lazy'
lazy -- Generation step 9 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy) --
+- Transformer.decode_step  target_len=9
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Decoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 9, 56)
Sampled token ID 53 -> 'vision'
vision -- Generation step 10 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision) --
+- Transformer.decode_step  target_len=10
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 10)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 10, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..9
|  |  |  [*] Positional encoding: shape=(10, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 10, 128)
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  [*] Decoder output: shape=(1, 10, 128)
|  +-  shape (1, 10, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 10, 56)
Sampled token ID 22 -> 'fast'
fast -- Generation step 11 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision fast) --
+- Transformer.decode_step  target_len=11
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 11)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 11, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..10
|  |  |  [*] Positional encoding: shape=(11, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 11, 128)
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  [*] Decoder output: shape=(1, 11, 128)
|  +-  shape (1, 11, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 11, 56)
Sampled token ID 16 -> 'computer'
computer -- Generation step 12 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision fast computer) --
+- Transformer.decode_step  target_len=12
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 12)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 12, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..11
|  |  |  [*] Positional encoding: shape=(12, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 12, 128)
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  [*] Decoder output: shape=(1, 12, 128)
|  +-  shape (1, 12, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 12, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 13 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision fast computer data) --
+- Transformer.decode_step  target_len=13
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 13)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 13, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..12
|  |  |  [*] Positional encoding: shape=(13, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 13, 128)
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  [*] Decoder output: shape=(1, 13, 128)
|  +-  shape (1, 13, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 13, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 14 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision fast computer data data) --
+- Transformer.decode_step  target_len=14
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 14)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 14, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..13
|  |  |  [*] Positional encoding: shape=(14, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 14, 128)
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  [*] Decoder output: shape=(1, 14, 128)
|  +-  shape (1, 14, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 14, 56)
Sampled token ID 33 -> 'love'
love -- Generation step 15 (Current input: <bos> intelligence translation translation data data <unk> <unk> lazy vision fast computer data data love)
--
+- Transformer.decode_step  target_len=15
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 15)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 15, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..14
|  |  |  [*] Positional encoding: shape=(15, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 15, 128)
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 12)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  [*] Decoder output: shape=(1, 15, 128)
|  +-  shape (1, 15, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 15, 56)
Sampled token ID 6 -> 'I'
I
--------------------------------------------------

Input : Artificial intelligence is transforming the world.
Output: -- Encoding source prompt: 'Artificial intelligence is transforming the world.' --
+- Tokenizer.encode  add_special=True
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "Artificial intelligence is transforming the world."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['Artificial', 'intelligence', 'is', 'transforming', 'the', 'world', '.']
|  +-  7 tokens
|  Token->ID mapping: [('Artificial', 5), ('intelligence', 26), ('is', 27), ('transforming', 51), ('the', 47), ('world', 55), ('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 5, 26, 27, 51, 47, 55, 4, 3]
+-  9 token IDs
+- Transformer.encode  encode source sequence only
|  Source mask: shape (1, 1, 1, 9)
|  +- TransformerEncoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Encoder Layer 1/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Encoder Layer 2/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Encoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
+-  encoder output shape (1, 9, 128)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 9)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 9)
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
Sampled token ID 33 -> 'love'
love -- Generation step 2 (Current input: <bos> love) --
+- Transformer.decode_step  target_len=2
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 2)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 2, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..1
|  |  |  [*] Positional encoding: shape=(2, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 2, 128)
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  [*] Decoder output: shape=(1, 2, 128)
|  +-  shape (1, 2, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 2, 56)
Sampled token ID 30 -> 'lazy'
lazy -- Generation step 3 (Current input: <bos> love lazy) --
+- Transformer.decode_step  target_len=3
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 3)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 3, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..2
|  |  |  [*] Positional encoding: shape=(3, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 3, 128)
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  [*] Decoder output: shape=(1, 3, 128)
|  +-  shape (1, 3, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 3, 56)
Sampled token ID 33 -> 'love'
love -- Generation step 4 (Current input: <bos> love lazy love) --
+- Transformer.decode_step  target_len=4
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 4)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 4, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..3
|  |  |  [*] Positional encoding: shape=(4, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 4, 128)
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  [*] Decoder output: shape=(1, 4, 128)
|  +-  shape (1, 4, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 4, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 5 (Current input: <bos> love lazy love data) --
+- Transformer.decode_step  target_len=5
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 5)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 5, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..4
|  |  |  [*] Positional encoding: shape=(5, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 5, 128)
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  [*] Decoder output: shape=(1, 5, 128)
|  +-  shape (1, 5, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 5, 56)
Sampled token ID 17 -> 'data'
data -- Generation step 6 (Current input: <bos> love lazy love data data) --
+- Transformer.decode_step  target_len=6
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 6)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 6, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..5
|  |  |  [*] Positional encoding: shape=(6, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 6, 128)
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  [*] Decoder output: shape=(1, 6, 128)
|  +-  shape (1, 6, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 6, 56)
Sampled token ID 54 -> 'weight'
weight -- Generation step 7 (Current input: <bos> love lazy love data data weight) --
+- Transformer.decode_step  target_len=7
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 7)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 7, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..6
|  |  |  [*] Positional encoding: shape=(7, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 7, 128)
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  [*] Decoder output: shape=(1, 7, 128)
|  +-  shape (1, 7, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 7, 56)
Sampled token ID 18 -> 'deep'
deep -- Generation step 8 (Current input: <bos> love lazy love data data weight deep) --
+- Transformer.decode_step  target_len=8
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 8)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 8, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..7
|  |  |  [*] Positional encoding: shape=(8, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 8, 128)
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  [*] Decoder output: shape=(1, 8, 128)
|  +-  shape (1, 8, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 8, 56)
Sampled token ID 27 -> 'is'
is -- Generation step 9 (Current input: <bos> love lazy love data data weight deep is) --
+- Transformer.decode_step  target_len=9
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Decoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 9, 56)
Sampled token ID 46 -> 'text'
text -- Generation step 10 (Current input: <bos> love lazy love data data weight deep is text) --
+- Transformer.decode_step  target_len=10
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 10)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 10, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..9
|  |  |  [*] Positional encoding: shape=(10, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 10, 128)
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  [*] Decoder output: shape=(1, 10, 128)
|  +-  shape (1, 10, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 10, 56)
Sampled token ID 31 -> 'learning'
learning -- Generation step 11 (Current input: <bos> love lazy love data data weight deep is text learning) --
+- Transformer.decode_step  target_len=11
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 11)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 11, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..10
|  |  |  [*] Positional encoding: shape=(11, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 11, 128)
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  [*] Decoder output: shape=(1, 11, 128)
|  +-  shape (1, 11, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 11, 56)
Sampled token ID 31 -> 'learning'
learning -- Generation step 12 (Current input: <bos> love lazy love data data weight deep is text learning learning) --
+- Transformer.decode_step  target_len=12
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 12)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 12, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..11
|  |  |  [*] Positional encoding: shape=(12, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 12, 128)
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  [*] Decoder output: shape=(1, 12, 128)
|  +-  shape (1, 12, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 12, 56)
Sampled token ID 6 -> 'I'
I -- Generation step 13 (Current input: <bos> love lazy love data data weight deep is text learning learning I) --
+- Transformer.decode_step  target_len=13
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 13)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 13, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..12
|  |  |  [*] Positional encoding: shape=(13, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 13, 128)
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  [*] Decoder output: shape=(1, 13, 128)
|  +-  shape (1, 13, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 13, 56)
Sampled token ID 31 -> 'learning'
learning -- Generation step 14 (Current input: <bos> love lazy love data data weight deep is text learning learning I learning) --
+- Transformer.decode_step  target_len=14
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 14)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 14, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..13
|  |  |  [*] Positional encoding: shape=(14, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 14, 128)
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  [*] Decoder output: shape=(1, 14, 128)
|  +-  shape (1, 14, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 14, 56)
Sampled token ID 38 -> 'quick'
quick -- Generation step 15 (Current input: <bos> love lazy love data data weight deep is text learning learning I learning quick) --
+- Transformer.decode_step  target_len=15
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 15)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 15, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..14
|  |  |  [*] Positional encoding: shape=(15, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 15, 128)
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  [*] Decoder output: shape=(1, 15, 128)
|  +-  shape (1, 15, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 15, 56)
Sampled token ID 46 -> 'text'
text
--------------------------------------------------

Input : I love learning about deep learning and transformers.
Output: -- Encoding source prompt: 'I love learning about deep learning and transformers.' --
+- Tokenizer.encode  add_special=True
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "I love learning about deep learning and transformers."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (9): ['I', 'love', 'learning', 'about', 'deep', 'learning', 'and', 'transformers', '.']
|  +-  9 tokens
|  Token->ID mapping: [('I', 6), ('love', 33), ('learning', 31), ('about', 9), ('deep', 18), ('learning', 31), ('and', 11), ('transformers', 50),
('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 6, 33, 31, 9, 18, 31, 11, 50, 4, 3]
+-  11 token IDs
+- Transformer.encode  encode source sequence only
|  Source mask: shape (1, 1, 1, 11)
|  +- TransformerEncoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 11)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 11, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..10
|  |  |  [*] Positional encoding: shape=(11, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 11, 128)
|  |  +-  shape (1, 11, 128)
|  |  -- Encoder Layer 1/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  -- Encoder Layer 2/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  [*] Encoder output: shape=(1, 11, 128)
|  +-  shape (1, 11, 128)
+-  encoder output shape (1, 11, 128)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 11)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 11)
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
Sampled token ID 2 -> '<bos>'
-- Generation step 2 (Current input: <bos> <bos>) --
+- Transformer.decode_step  target_len=2
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 2)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 2, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..1
|  |  |  [*] Positional encoding: shape=(2, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 2, 128)
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  [*] Decoder output: shape=(1, 2, 128)
|  +-  shape (1, 2, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 2, 56)
Sampled token ID 39 -> 'robot'
robot -- Generation step 3 (Current input: <bos> <bos> robot) --
+- Transformer.decode_step  target_len=3
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 3)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 3, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..2
|  |  |  [*] Positional encoding: shape=(3, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 3, 128)
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  [*] Decoder output: shape=(1, 3, 128)
|  +-  shape (1, 3, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 3, 56)
Sampled token ID 26 -> 'intelligence'
intelligence -- Generation step 4 (Current input: <bos> <bos> robot intelligence) --
+- Transformer.decode_step  target_len=4
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 4)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 4, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..3
|  |  |  [*] Positional encoding: shape=(4, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 4, 128)
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  [*] Decoder output: shape=(1, 4, 128)
|  +-  shape (1, 4, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 4, 56)
Sampled token ID 39 -> 'robot'
robot -- Generation step 5 (Current input: <bos> <bos> robot intelligence robot) --
+- Transformer.decode_step  target_len=5
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 5)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 5, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..4
|  |  |  [*] Positional encoding: shape=(5, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 5, 128)
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  [*] Decoder output: shape=(1, 5, 128)
|  +-  shape (1, 5, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 5, 56)
Sampled token ID 25 -> 'future'
future -- Generation step 6 (Current input: <bos> <bos> robot intelligence robot future) --
+- Transformer.decode_step  target_len=6
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 6)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 6, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..5
|  |  |  [*] Positional encoding: shape=(6, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 6, 128)
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  [*] Decoder output: shape=(1, 6, 128)
|  +-  shape (1, 6, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 6, 56)
Sampled token ID 49 -> 'to'
to -- Generation step 7 (Current input: <bos> <bos> robot intelligence robot future to) --
+- Transformer.decode_step  target_len=7
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 7)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 7, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..6
|  |  |  [*] Positional encoding: shape=(7, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 7, 128)
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  [*] Decoder output: shape=(1, 7, 128)
|  +-  shape (1, 7, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 7, 56)
Sampled token ID 44 -> 'speech'
speech -- Generation step 8 (Current input: <bos> <bos> robot intelligence robot future to speech) --
+- Transformer.decode_step  target_len=8
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 8)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 8, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..7
|  |  |  [*] Positional encoding: shape=(8, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 8, 128)
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  [*] Decoder output: shape=(1, 8, 128)
|  +-  shape (1, 8, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 8, 56)
Sampled token ID 15 -> 'code'
code -- Generation step 9 (Current input: <bos> <bos> robot intelligence robot future to speech code) --
+- Transformer.decode_step  target_len=9
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Decoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 9, 56)
Sampled token ID 4 -> '.'
. -- Generation step 10 (Current input: <bos> <bos> robot intelligence robot future to speech code .) --
+- Transformer.decode_step  target_len=10
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 10)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 10, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..9
|  |  |  [*] Positional encoding: shape=(10, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 10, 128)
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  [*] Decoder output: shape=(1, 10, 128)
|  +-  shape (1, 10, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 10, 56)
Sampled token ID 4 -> '.'
. -- Generation step 11 (Current input: <bos> <bos> robot intelligence robot future to speech code . .) --
+- Transformer.decode_step  target_len=11
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 11)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 11, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..10
|  |  |  [*] Positional encoding: shape=(11, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 11, 128)
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  [*] Decoder output: shape=(1, 11, 128)
|  +-  shape (1, 11, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 11, 56)
Sampled token ID 15 -> 'code'
code -- Generation step 12 (Current input: <bos> <bos> robot intelligence robot future to speech code . . code) --
+- Transformer.decode_step  target_len=12
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 12)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 12, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..11
|  |  |  [*] Positional encoding: shape=(12, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 12, 128)
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  [*] Decoder output: shape=(1, 12, 128)
|  +-  shape (1, 12, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 12, 56)
Sampled token ID 15 -> 'code'
code -- Generation step 13 (Current input: <bos> <bos> robot intelligence robot future to speech code . . code code) --
+- Transformer.decode_step  target_len=13
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 13)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 13, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..12
|  |  |  [*] Positional encoding: shape=(13, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 13, 128)
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 11)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  [*] Decoder output: shape=(1, 13, 128)
|  +-  shape (1, 13, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 13, 56)
Sampled token ID 3 -> '<eos>'
<eos>Reached <eos> token, stopping generation.

--------------------------------------------------

Input : Translate this sentence to another language.
Output: -- Encoding source prompt: 'Translate this sentence to another language.' --
+- Tokenizer.encode  add_special=True
|  +- Tokenizer.tokenize  lang='en'
|  |  Input text: "Translate this sentence to another language."
|  |  Using spaCy rule-based tokenizer
|  |  Tokens (7): ['Translate', 'this', 'sentence', 'to', 'another', 'language', '.']
|  +-  7 tokens
|  Token->ID mapping: [('Translate', 8), ('this', 48), ('sentence', 42), ('to', 49), ('another', 12), ('language', 29), ('.', 4)]
|  With special tokens: [<bos>=2] + ids + [<eos>=3]
|  Final IDs: [2, 8, 48, 42, 49, 12, 29, 4, 3]
+-  9 token IDs
+- Transformer.encode  encode source sequence only
|  Source mask: shape (1, 1, 1, 9)
|  +- TransformerEncoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Encoder Layer 1/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Encoder Layer 2/2 --
|  |  +- EncoderLayer  dim=128
|  |  |  -- Sub-layer 1: Self-Attention --
|  |  |  Q = K = V = x  (each token attends to all tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + Attention(x))
|  |  |  -- Sub-layer 2: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Encoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
+-  encoder output shape (1, 9, 128)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 9)
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
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 1, 9)
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
Sampled token ID 15 -> 'code'
code -- Generation step 2 (Current input: <bos> code) --
+- Transformer.decode_step  target_len=2
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 2)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 2, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..1
|  |  |  [*] Positional encoding: shape=(2, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 2, 128)
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 2, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 2)
|  |  |  |  |  Mask applied: 1 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 2, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 2, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 2, 32)
|  |  |  |  +-  shape (1, 4, 2, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 2, 128)
|  |  |  +-  shape (1, 2, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 2, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 2, 128)
|  |  [*] Decoder output: shape=(1, 2, 128)
|  +-  shape (1, 2, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 2, 56)
Sampled token ID 41 -> 'science'
science -- Generation step 3 (Current input: <bos> code science) --
+- Transformer.decode_step  target_len=3
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 3)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 3, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..2
|  |  |  [*] Positional encoding: shape=(3, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 3, 128)
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 3, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 3)
|  |  |  |  |  Mask applied: 3 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 3, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 3, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 3, 32)
|  |  |  |  +-  shape (1, 4, 3, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 3, 128)
|  |  |  +-  shape (1, 3, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 3, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 3, 128)
|  |  [*] Decoder output: shape=(1, 3, 128)
|  +-  shape (1, 3, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 3, 56)
Sampled token ID 44 -> 'speech'
speech -- Generation step 4 (Current input: <bos> code science speech) --
+- Transformer.decode_step  target_len=4
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 4)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 4, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..3
|  |  |  [*] Positional encoding: shape=(4, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 4, 128)
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 4, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 4)
|  |  |  |  |  Mask applied: 6 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 4, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 4, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 4, 32)
|  |  |  |  +-  shape (1, 4, 4, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 4, 128)
|  |  |  +-  shape (1, 4, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 4, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 4, 128)
|  |  [*] Decoder output: shape=(1, 4, 128)
|  +-  shape (1, 4, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 4, 56)
Sampled token ID 35 -> 'neural'
neural -- Generation step 5 (Current input: <bos> code science speech neural) --
+- Transformer.decode_step  target_len=5
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 5)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 5, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..4
|  |  |  [*] Positional encoding: shape=(5, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 5, 128)
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 5, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 5)
|  |  |  |  |  Mask applied: 10 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 5, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 5, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 5, 32)
|  |  |  |  +-  shape (1, 4, 5, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 5, 128)
|  |  |  +-  shape (1, 5, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 5, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 5, 128)
|  |  [*] Decoder output: shape=(1, 5, 128)
|  +-  shape (1, 5, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 5, 56)
Sampled token ID 42 -> 'sentence'
sentence -- Generation step 6 (Current input: <bos> code science speech neural sentence) --
+- Transformer.decode_step  target_len=6
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 6)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 6, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..5
|  |  |  [*] Positional encoding: shape=(6, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 6, 128)
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 6, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 6)
|  |  |  |  |  Mask applied: 15 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 6, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 6, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 6, 32)
|  |  |  |  +-  shape (1, 4, 6, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 6, 128)
|  |  |  +-  shape (1, 6, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 6, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 6, 128)
|  |  [*] Decoder output: shape=(1, 6, 128)
|  +-  shape (1, 6, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 6, 56)
Sampled token ID 50 -> 'transformers'
transformers -- Generation step 7 (Current input: <bos> code science speech neural sentence transformers) --
+- Transformer.decode_step  target_len=7
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 7)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 7, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..6
|  |  |  [*] Positional encoding: shape=(7, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 7, 128)
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 7, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 7)
|  |  |  |  |  Mask applied: 21 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 7, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 7, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 7, 32)
|  |  |  |  +-  shape (1, 4, 7, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 7, 128)
|  |  |  +-  shape (1, 7, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 7, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 7, 128)
|  |  [*] Decoder output: shape=(1, 7, 128)
|  +-  shape (1, 7, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 7, 56)
Sampled token ID 15 -> 'code'
code -- Generation step 8 (Current input: <bos> code science speech neural sentence transformers code) --
+- Transformer.decode_step  target_len=8
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 8)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 8, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..7
|  |  |  [*] Positional encoding: shape=(8, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 8, 128)
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 8, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 8)
|  |  |  |  |  Mask applied: 28 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 8, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 8, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 8, 32)
|  |  |  |  +-  shape (1, 4, 8, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 8, 128)
|  |  |  +-  shape (1, 8, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 8, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 8, 128)
|  |  [*] Decoder output: shape=(1, 8, 128)
|  +-  shape (1, 8, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 8, 56)
Sampled token ID 13 -> 'bias'
bias -- Generation step 9 (Current input: <bos> code science speech neural sentence transformers code bias) --
+- Transformer.decode_step  target_len=9
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 9)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 9, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..8
|  |  |  [*] Positional encoding: shape=(9, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 9, 128)
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 36 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 9, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 9, 32)
|  |  |  |  +-  shape (1, 4, 9, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 9, 128)
|  |  |  +-  shape (1, 9, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 9, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 9, 128)
|  |  [*] Decoder output: shape=(1, 9, 128)
|  +-  shape (1, 9, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 9, 56)
Sampled token ID 55 -> 'world'
world -- Generation step 10 (Current input: <bos> code science speech neural sentence transformers code bias world) --
+- Transformer.decode_step  target_len=10
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 10)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 10, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..9
|  |  |  [*] Positional encoding: shape=(10, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 10, 128)
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 10, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 10)
|  |  |  |  |  Mask applied: 45 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 10, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 10, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 10, 32)
|  |  |  |  +-  shape (1, 4, 10, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 10, 128)
|  |  |  +-  shape (1, 10, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 10, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 10, 128)
|  |  [*] Decoder output: shape=(1, 10, 128)
|  +-  shape (1, 10, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 10, 56)
Sampled token ID 5 -> 'Artificial'
Artificial -- Generation step 11 (Current input: <bos> code science speech neural sentence transformers code bias world Artificial) --
+- Transformer.decode_step  target_len=11
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 11)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 11, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..10
|  |  |  [*] Positional encoding: shape=(11, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 11, 128)
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 11, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 11)
|  |  |  |  |  Mask applied: 55 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 11, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 11, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 11, 32)
|  |  |  |  +-  shape (1, 4, 11, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 11, 128)
|  |  |  +-  shape (1, 11, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 11, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 11, 128)
|  |  [*] Decoder output: shape=(1, 11, 128)
|  +-  shape (1, 11, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 11, 56)
Sampled token ID 5 -> 'Artificial'
Artificial -- Generation step 12 (Current input: <bos> code science speech neural sentence transformers code bias world Artificial Artificial) --
+- Transformer.decode_step  target_len=12
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 12)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 12, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..11
|  |  |  [*] Positional encoding: shape=(12, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 12, 128)
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 12, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 12)
|  |  |  |  |  Mask applied: 66 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 12, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 12, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 12, 32)
|  |  |  |  +-  shape (1, 4, 12, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 12, 128)
|  |  |  +-  shape (1, 12, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 12, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 12, 128)
|  |  [*] Decoder output: shape=(1, 12, 128)
|  +-  shape (1, 12, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 12, 56)
Sampled token ID 42 -> 'sentence'
sentence -- Generation step 13 (Current input: <bos> code science speech neural sentence transformers code bias world Artificial Artificial sentence) --
+- Transformer.decode_step  target_len=13
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 13)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 13, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..12
|  |  |  [*] Positional encoding: shape=(13, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 13, 128)
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 13, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 13)
|  |  |  |  |  Mask applied: 78 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 13, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 13, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 13, 32)
|  |  |  |  +-  shape (1, 4, 13, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 13, 128)
|  |  |  +-  shape (1, 13, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 13, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 13, 128)
|  |  [*] Decoder output: shape=(1, 13, 128)
|  +-  shape (1, 13, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 13, 56)
Sampled token ID 42 -> 'sentence'
sentence -- Generation step 14 (Current input: <bos> code science speech neural sentence transformers code bias world Artificial Artificial sentence
sentence) --
+- Transformer.decode_step  target_len=14
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 14)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 14, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..13
|  |  |  [*] Positional encoding: shape=(14, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 14, 128)
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 14, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 14)
|  |  |  |  |  Mask applied: 91 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 14, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 14, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 14, 32)
|  |  |  |  +-  shape (1, 4, 14, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 14, 128)
|  |  |  +-  shape (1, 14, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 14, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 14, 128)
|  |  [*] Decoder output: shape=(1, 14, 128)
|  +-  shape (1, 14, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 14, 56)
Sampled token ID 38 -> 'quick'
quick -- Generation step 15 (Current input: <bos> code science speech neural sentence transformers code bias world Artificial Artificial sentence
sentence quick) --
+- Transformer.decode_step  target_len=15
|  +- TransformerDecoder  2 layers
|  |  -- Embedding --
|  |  +- TransformerEmbedding  vocab_size=56, dim=128
|  |  |  [*] Input token IDs: shape=(1, 15)
|  |  |  TokenEmbedding: lookup table -> dense vectors, * sqrt(128) = * 11.31 scaling
|  |  |  [*] Token embeddings (scaled): shape=(1, 15, 128)
|  |  |  PositionalEncoding: sin/cos signals for positions 0..14
|  |  |  [*] Positional encoding: shape=(15, 128)
|  |  |  Combined: token_embeddings + positional_encoding -> dropout
|  |  |  [*] Output embeddings: shape=(1, 15, 128)
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 1/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  -- Decoder Layer 2/2 --
|  |  +- DecoderLayer  dim=128
|  |  |  -- Sub-layer 1: Masked Self-Attention --
|  |  |  Q = K = V = x  (causal mask prevents seeing future tokens)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 15, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 15)
|  |  |  |  |  Mask applied: 105 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + MaskedAttn(x))
|  |  |  -- Sub-layer 2: Cross-Attention --
|  |  |  Q = decoder_x, K = V = encoder_output  (reading source sentence)
|  |  |  +- MultiHeadAttention  4 heads * 32 dims
|  |  |  |  Linear projections -> split into 4 heads
|  |  |  |  [*] Q (per head): shape=(1, 4, 15, 32)
|  |  |  |  [*] K (per head): shape=(1, 4, 9, 32)
|  |  |  |  [*] V (per head): shape=(1, 4, 9, 32)
|  |  |  |  +- ScaledDotProductAttention  head_dim=32
|  |  |  |  |  scores = Q * K^T / sqrt(32) = Q * K^T / 5.66
|  |  |  |  |  [*] Attention scores: shape=(1, 4, 15, 9)
|  |  |  |  |  Mask applied: 0 positions masked to -inf
|  |  |  |  |  softmax -> attention weights (probabilities summing to 1)
|  |  |  |  |  [*] Context vectors: shape=(1, 4, 15, 32)
|  |  |  |  +-  shape (1, 4, 15, 32)
|  |  |  |  Merge 4 heads -> concatenate -> output projection
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + CrossAttn(x, enc))
|  |  |  -- Sub-layer 3: Feed-Forward --
|  |  |  +- FeedForward  128 -> 512 -> 128
|  |  |  |  Linear -> ReLU -> Dropout -> Linear
|  |  |  |  [*] Output: shape=(1, 15, 128)
|  |  |  +-  shape (1, 15, 128)
|  |  |  LayerNorm: normalize features -> mean~=0, std~=1 -> shape (1, 15, 128)
|  |  |  Residual connection: x = LayerNorm(x + FFN(x))
|  |  +-  shape (1, 15, 128)
|  |  [*] Decoder output: shape=(1, 15, 128)
|  +-  shape (1, 15, 128)
|  Output projection: 128 -> 56 (vocab logits)
+-  logits shape (1, 15, 56)
Sampled token ID 15 -> 'code'
code
--------------------------------------------------


Main Menu
1. Trace Embedding Pipeline (Step-by-step visualization)
2. View Model Architecture (Summary table)
3. Text Generation Demo (Greedy decoding process)
4. Toggle Deep Trace Mode (Currently: ON)
5. Exit
Select an option [1/2/3/4/5]: 5
Goodbye!
(venv)

```


## 📖 Reference

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). **Attention is all you need.** *Advances in neural information processing systems*, 30.
