import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import trace

class ScaledDotProductAttention(nn.Module):
    """
    Computes the 'scaled dot product' attention between query, key, and value tensors.
    
    The 'scaled' part (dividing by sqrt of d_k) prevents the gradients from 
    vanishing during the softmax step for large embedding dimensions.
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q, k, v shape: (batch_size, num_heads, seq_length, head_dim)
        head_dim = q.size(-1)
        trace.enter("ScaledDotProductAttention", f"head_dim={head_dim}")
        
        # 1. Calculate similarity scores
        # scores shape: (batch_size, num_heads, seq_length_q, seq_length_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        trace.log(f"scores = Q * K^T / sqrt({head_dim}) = Q * K^T / {math.sqrt(head_dim):.2f}")
        trace.tensor("Attention scores", scores)
        
        # 2. Apply mask (e.g., to ignore padding or future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
            trace.log(f"Mask applied: {mask.sum().item():.0f} positions masked to -inf")
            
        # 3. Convert scores to probabilities
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        trace.log("softmax -> attention weights (probabilities summing to 1)")
        
        # 4. Use weights to get a weighted sum of values
        context_vectors = torch.matmul(attention_weights, v)
        trace.tensor("Context vectors", context_vectors)
        trace.exit(summary=f"shape {tuple(context_vectors.shape)}")
        
        return context_vectors, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Projections for Query, Key, and Value
        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        
        # Final output projection
        self.out_linear = nn.Linear(embedding_dim, embedding_dim)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        trace.enter("MultiHeadAttention", f"{self.num_heads} heads * {self.head_dim} dims")
        
        # 1. Project input to Q, K, V and split into multiple heads
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))
        trace.log(f"Linear projections -> split into {self.num_heads} heads")
        trace.tensor("Q (per head)", q)
        trace.tensor("K (per head)", k)
        trace.tensor("V (per head)", v)
        
        # 2. Apply scaled dot product attention on each head
        context, self.attention_weights = self.attention(q, k, v, mask=mask)
        
        # 3. Concatenate (merge) heads back and apply final linear layer
        output = self.merge_heads(context)
        trace.log(f"Merge {self.num_heads} heads -> concatenate -> output projection")
        result = self.out_linear(output)
        trace.tensor("Output", result)
        trace.exit(summary=f"shape {tuple(result.shape)}")
        return result


class PositionWiseFeedForward(nn.Module):
    """
    A simple fully connected feed-forward network, which is applied to each 
    position identically and independently.
    """
    def __init__(self, embedding_dim: int, feed_forward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feed_forward_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trace.enter("FeedForward", f"{x.shape[-1]} -> {self.net[0].out_features} -> {x.shape[-1]}")
        trace.log("Linear -> ReLU -> Dropout -> Linear")
        result = self.net(x)
        trace.tensor("Output", result)
        trace.exit(summary=f"shape {tuple(result.shape)}")
        return result

class LayerNormalization(nn.Module):
    """
    Normalizes the inputs across the features dimension for each token.
    """
    def __init__(self, embedding_dim: int, epsilon: float = 1e-12):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        result = self.alpha * (x - mean) / (std + self.epsilon) + self.bias
        trace.log(f"LayerNorm: normalize features -> mean~=0, std~=1 -> shape {tuple(result.shape)}", style="dim")
        return result

