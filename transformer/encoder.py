import torch
import torch.nn as nn
from .embeddings import TransformerEmbedding
from .layers import MultiHeadAttention, PositionWiseFeedForward, LayerNormalization
from . import trace

class EncoderLayer(nn.Module):
    """
    A single layer of the Encoder.
    
    It consists of two main sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network
    Each sub-layer is followed by layer normalization and has a residual connection.
    """
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, feed_forward_dim, dropout)
        
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        trace.enter("EncoderLayer", f"dim={x.shape[-1]}")
        
        # 1. Self-Attention + Residual + LayerNorm
        trace.divider("Sub-layer 1: Self-Attention")
        trace.log("Q = K = V = x  (each token attends to all tokens)")
        attention_output = self.self_attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attention_output))
        trace.log("Residual connection: x = LayerNorm(x + Attention(x))")
        
        # 2. Feed-Forward + Residual + LayerNorm
        trace.divider("Sub-layer 2: Feed-Forward")
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        trace.log("Residual connection: x = LayerNorm(x + FFN(x))")
        
        trace.exit(summary=f"shape {tuple(x.shape)}")
        return x

class TransformerEncoder(nn.Module):
    """
    The full Encoder stack.
    
    Stacks multiple EncoderLayers on top of the initial embedding layer.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        num_layers: int, 
        num_heads: int, 
        feed_forward_dim: int, 
        max_seq_length: int = 5000, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embedding_dim, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, feed_forward_dim, dropout) 
            for _ in range(num_layers)
        ])

    def build_source_mask(self, source_tokens: torch.Tensor, pad_index: int = 0) -> torch.Tensor:
        """
        Creates a mask to ignore padding tokens in the source sequence.
        """
        # (batch_size, 1, 1, seq_length)
        return (source_tokens == pad_index).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        trace.enter("TransformerEncoder", f"{len(self.layers)} layers")
        
        # Apply initial embedding
        trace.divider("Embedding")
        x = self.embedding(x)
        
        # Pass through each encoder layer
        for i, layer in enumerate(self.layers):
            trace.divider(f"Encoder Layer {i+1}/{len(self.layers)}")
            x = layer(x, mask)
        
        trace.tensor("Encoder output", x)
        trace.exit(summary=f"shape {tuple(x.shape)}")
        return x
