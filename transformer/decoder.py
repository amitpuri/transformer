import torch
import torch.nn as nn
from .embeddings import TransformerEmbedding
from .layers import MultiHeadAttention, PositionWiseFeedForward, LayerNormalization
from . import trace

class DecoderLayer(nn.Module):
    """
    A single layer of the Decoder.
    
    It consists of three main sub-layers:
    1. Masked Multi-Head Self-Attention (prevents looking at future tokens)
    2. Multi-Head Cross-Attention (attends to the encoder's output)
    3. Position-wise Feed-Forward Network
    """
    def __init__(self, embedding_dim: int, num_heads: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(embedding_dim, feed_forward_dim, dropout)
        
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
        self.norm3 = LayerNormalization(embedding_dim)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, target_mask: torch.Tensor = None, source_mask: torch.Tensor = None) -> torch.Tensor:
        trace.enter("DecoderLayer", f"dim={x.shape[-1]}")
        
        # 1. Masked Self-Attention + Residual + LayerNorm
        trace.divider("Sub-layer 1: Masked Self-Attention")
        trace.log("Q = K = V = x  (causal mask prevents seeing future tokens)")
        self_attn_out = self.self_attention(x, x, x, mask=target_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        trace.log("Residual connection: x = LayerNorm(x + MaskedAttn(x))")
        
        # 2. Cross-Attention + Residual + LayerNorm
        # Query comes from decoder (x), Key and Value come from encoder_output
        trace.divider("Sub-layer 2: Cross-Attention")
        trace.log("Q = decoder_x, K = V = encoder_output  (reading source sentence)")
        cross_attn_out = self.cross_attention(x, encoder_output, encoder_output, mask=source_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        trace.log("Residual connection: x = LayerNorm(x + CrossAttn(x, enc))")
        
        # 3. Feed-Forward + Residual + LayerNorm
        trace.divider("Sub-layer 3: Feed-Forward")
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        trace.log("Residual connection: x = LayerNorm(x + FFN(x))")
        
        trace.exit(summary=f"shape {tuple(x.shape)}")
        return x

class TransformerDecoder(nn.Module):
    """
    The full Decoder stack.
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
            DecoderLayer(embedding_dim, num_heads, feed_forward_dim, dropout) 
            for _ in range(num_layers)
        ])

    def build_target_mask(self, target_tokens: torch.Tensor, pad_index: int = 0) -> torch.Tensor:
        """
        Creates a combined mask: 
        1. Padding mask (ignore pad tokens)
        2. Look-ahead mask (prevent attending to future tokens)
        """
        batch_size, seq_length = target_tokens.size()
        
        # 1. Padding mask: (batch_size, 1, 1, seq_length)
        padding_mask = (target_tokens == pad_index).unsqueeze(1).unsqueeze(2)
        
        # 2. Look-ahead mask: (1, 1, seq_length, seq_length)
        # Creates an upper triangular matrix of ones, then converts to boolean
        look_ahead_mask = torch.triu(torch.ones((seq_length, seq_length), device=target_tokens.device), diagonal=1).bool()
        
        # Combine them (logical OR)
        return padding_mask | look_ahead_mask

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, target_mask: torch.Tensor = None, source_mask: torch.Tensor = None) -> torch.Tensor:
        trace.enter("TransformerDecoder", f"{len(self.layers)} layers")
        
        # Apply initial embedding
        trace.divider("Embedding")
        x = self.embedding(x)
        
        # Pass through each decoder layer
        for i, layer in enumerate(self.layers):
            trace.divider(f"Decoder Layer {i+1}/{len(self.layers)}")
            x = layer(x, encoder_output, target_mask, source_mask)
        
        trace.tensor("Decoder output", x)
        trace.exit(summary=f"shape {tuple(x.shape)}")
        return x
