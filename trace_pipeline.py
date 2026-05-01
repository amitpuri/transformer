import math
import torch
from rich.console import Console
from rich.table import Table
from transformer.tokenization import SpacyTokenizer
from transformer.embeddings import TransformerEmbedding

console = Console()

def format_vector(tensor_slice: torch.Tensor, decimals: int = 3) -> list:
    return [round(n.item(), decimals) for n in tensor_slice]

def run_trace_demo(text: str = "I like to learn about AI."):
    console.print(f"\n[bold green]Tracing Embedding Pipeline for:[/bold green] '{text}'\n")

    # 1. Tokenization
    tokenizer = SpacyTokenizer()
    tokens = tokenizer.tokenize(text)
    console.print(f"[bold]Step 1: Tokenization[/bold] -> {tokens}")

    # 2. Vocabulary & Encoding
    tokenizer.build_vocab([text], lang="en", min_freq=1)
    vocab = tokenizer.vocabs["en"]
    ids = tokenizer.encode(text, lang="en", add_special=True)
    all_tokens = ["<bos>"] + tokens + ["<eos>"]
    
    table = Table(title="Step 2: Vocabulary Mapping")
    table.add_column("Token", style="cyan")
    table.add_column("ID", style="magenta")
    for tok, id_ in zip(all_tokens, ids):
        table.add_row(tok, str(id_))
    console.print(table)

    # 3. Embedding
    embedding_dim = 64
    emb_layer = TransformerEmbedding(len(vocab), embedding_dim)
    emb_layer.eval()
    
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        final_embeddings = emb_layer(x)
    
    console.print(f"\n[bold]Step 3: Multi-dimensional Representation[/bold]")
    console.print(f"Final shape: [yellow]{tuple(final_embeddings.shape)}[/yellow] (Batch, Seq, Dim)")
    
    # Show samples
    sample_table = Table(title="Final Embedding Samples (First 4 dims)")
    sample_table.add_column("Token")
    sample_table.add_column("Embedding Vector")
    for i, tok in enumerate(all_tokens):
        vec = format_vector(final_embeddings[0, i, :4])
        sample_table.add_row(tok, str(vec))
    console.print(sample_table)

if __name__ == "__main__":
    run_trace_demo()
