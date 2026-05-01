import math
import torch
import torch.nn as nn
from . import trace

class TokenEmbedding(nn.Embedding):
    """
    Maps token indices to dense vectors of a specified dimension.
    
    This layer learns a continuous representation for each word in the vocabulary.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        # padding_idx=0 ensures that the padding token remains a zero vector.
        super().__init__(vocab_size, embedding_dim, padding_idx=0)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects position information into token embeddings using sine and cosine functions.
    
    Since Transformers process all tokens in parallel, they need this 'signal' 
    to understand the order of words in a sentence.
    """
    def __init__(self, embedding_dim: int, max_seq_length: int = 5000):
        super().__init__()
        
        # Create a matrix of [max_seq_length, embedding_dim] representing positional signals
        pe = torch.zeros(max_seq_length, embedding_dim)
        pe.requires_grad = False
        
        position = torch.arange(0, max_seq_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as a buffer (stays with the model but isn't a learned parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length)
        return self.pe[:x.size(1), :]

class TransformerEmbedding(nn.Module):
    """
    The complete embedding layer: Token Embedding + Positional Encoding.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embedding_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embedding_dim, max_seq_length)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trace.enter("TransformerEmbedding", f"vocab_size={self.token_emb.num_embeddings}, dim={self.embedding_dim}")
        trace.tensor("Input token IDs", x)
        
        # 1. Convert IDs to vectors and scale by sqrt of dimension (standard Transformer practice)
        tokens = self.token_emb(x) * math.sqrt(self.embedding_dim)
        trace.log(f"TokenEmbedding: lookup table -> dense vectors, * sqrt({self.embedding_dim}) = * {math.sqrt(self.embedding_dim):.2f} scaling")
        trace.tensor("Token embeddings (scaled)", tokens)
        
        # 2. Add positional signals
        positions = self.pos_enc(x)
        trace.log(f"PositionalEncoding: sin/cos signals for positions 0..{x.size(1)-1}")
        trace.tensor("Positional encoding", positions)
        
        # 3. Combine and apply dropout
        combined = self.dropout(tokens + positions)
        trace.log("Combined: token_embeddings + positional_encoding -> dropout")
        trace.tensor("Output embeddings", combined)
        trace.exit(summary=f"shape {tuple(combined.shape)}")
        return combined
