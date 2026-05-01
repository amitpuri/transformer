import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from . import trace

class Transformer(nn.Module):
    """
    The full Transformer architecture as described in 'Attention Is All You Need'.
    
    This model combines an Encoder and a Decoder with an output linear projection.
    """
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        feed_forward_dim: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1,
        pad_index: int = 0
    ):
        super().__init__()
        self.pad_index = pad_index
        
        self.encoder = TransformerEncoder(
            source_vocab_size, embedding_dim, num_layers, num_heads, feed_forward_dim, max_seq_length, dropout
        )
        
        self.decoder = TransformerDecoder(
            target_vocab_size, embedding_dim, num_layers, num_heads, feed_forward_dim, max_seq_length, dropout
        )
        
        # Final linear layer to project decoder outputs to vocabulary size
        self.output_projection = nn.Linear(embedding_dim, target_vocab_size)
        
        # Weight Tying: share weights between decoder embedding and output projection
        # This is a common trick to reduce parameters and improve performance.
        self.output_projection.weight = self.decoder.embedding.token_emb.weight
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Xavier Uniform initialization to all parameters with more than one dimension.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor, source_mask: torch.Tensor = None, target_mask: torch.Tensor = None):
        trace.enter("Transformer.forward")
        trace.tensor("Source tokens", source_tokens)
        trace.tensor("Target tokens", target_tokens)
        
        # 1. Build masks if they aren't provided
        trace.divider("Step 1: Build Masks")
        if source_mask is None:
            source_mask = self.encoder.build_source_mask(source_tokens, self.pad_index)
            trace.log(f"Source padding mask: shape {tuple(source_mask.shape)}")
        if target_mask is None:
            target_mask = self.decoder.build_target_mask(target_tokens, self.pad_index)
            trace.log(f"Target causal mask: shape {tuple(target_mask.shape)} (padding + look-ahead)")
            
        # 2. Encode the source sequence
        trace.divider("Step 2: Encode source sequence")
        encoder_output = self.encoder(source_tokens, source_mask)
        
        # 3. Decode the target sequence using the encoder output as context
        trace.divider("Step 3: Decode target sequence")
        decoder_output = self.decoder(target_tokens, encoder_output, target_mask, source_mask)
        
        # 4. Project to vocabulary logits
        trace.divider("Step 4: Output projection")
        logits = self.output_projection(decoder_output)
        trace.log(f"Linear({decoder_output.shape[-1]} -> {logits.shape[-1]}): decoder hidden -> vocab logits")
        trace.tensor("Logits", logits)
        
        trace.exit(summary=f"logits shape {tuple(logits.shape)}")
        return logits

    def encode(self, source_tokens: torch.Tensor, source_mask: torch.Tensor = None):
        """
        Helper for inference: runs just the encoder.
        """
        trace.enter("Transformer.encode", "encode source sequence only")
        if source_mask is None:
            source_mask = self.encoder.build_source_mask(source_tokens, self.pad_index)
            trace.log(f"Source mask: shape {tuple(source_mask.shape)}")
        result = self.encoder(source_tokens, source_mask)
        trace.exit(summary=f"encoder output shape {tuple(result.shape)}")
        return result, source_mask

    def decode_step(self, target_tokens: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor = None):
        """
        Helper for inference: runs a single decoding step.
        """
        trace.enter("Transformer.decode_step", f"target_len={target_tokens.shape[-1]}")
        if target_mask is None:
            target_mask = self.decoder.build_target_mask(target_tokens, self.pad_index)
        decoder_output = self.decoder(target_tokens, encoder_output, target_mask, source_mask)
        logits = self.output_projection(decoder_output)
        trace.log(f"Output projection: {decoder_output.shape[-1]} -> {logits.shape[-1]} (vocab logits)")
        trace.exit(summary=f"logits shape {tuple(logits.shape)}")
        return logits
