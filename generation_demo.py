import torch
from rich.console import Console
from rich.live import Live
from rich.table import Table
from transformer.model import Transformer
from transformer.tokenization import SpacyTokenizer
from transformer import trace

console = Console()

def sample_decode(model, tokenizer, prompt, max_len=15, temperature=1.2, lang="en"):
    model.eval()
    
    # 1. Encode source
    trace.divider(f"Encoding source prompt: '{prompt}'")
    ids = tokenizer.encode(prompt, lang=lang, add_special=True)
    src_tensor = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad():
        enc_out, src_mask = model.encode(src_tensor)
        
    vocab = tokenizer.vocabs[lang]
    generated_ids = [vocab.bos_id]
    tokens = ["<bos>"]
    
    for i in range(max_len):
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long)
        
        trace.divider(f"Generation step {i+1} (Current input: {' '.join(tokens)})")
        with torch.no_grad():
            logits = model.decode_step(tgt_tensor, enc_out, src_mask)
            
        next_token_logits = logits[0, -1, :] / temperature
        
        # To avoid early stopping in the demo, we'll slightly penalize <eos>
        # for the first few tokens.
        if i < 5:
            next_token_logits[vocab.eos_id] -= 10.0
            
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        generated_ids.append(next_id)
        token = vocab.to_token(next_id)
        trace.log(f"Sampled token ID {next_id} -> '{token}'")
        tokens.append(token)
        
        yield tokens
        
        if next_id == vocab.eos_id:
            trace.log("Reached <eos> token, stopping generation.")
            break

def run_generation_demo():
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "I love learning about deep learning and transformers.",
        "Translate this sentence to another language."
    ]
    
    console.print("\n[bold green]Text Generation Demo[/bold green]")
    console.print("[dim]Note: The model is untrained (random weights).[/dim]")
    console.print("[dim]Using sampling to visualize the decoding process.[/dim]\n")
    
    # Enrich vocabulary with many common words to ensure variety
    more_words = [
        "network", "neuron", "weight", "bias", "activation", "function", "data", "science",
        "computer", "vision", "speech", "text", "translation", "robot", "future", "digital",
        "spark", "logic", "dream", "code", "run", "fast", "deep", "neural", "system"
    ]
    tokenizer = SpacyTokenizer()
    tokenizer.build_vocab(prompts + [" ".join(more_words)], lang="en", min_freq=1)
    vocab_size = len(tokenizer.vocabs["en"])
    
    model = Transformer(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=512
    )
    
    for prompt in prompts:
        console.print(f"[bold blue]Input :[/bold blue] {prompt}")
        console.print("[bold yellow]Output:[/bold yellow] ", end="")
        
        for tokens in sample_decode(model, tokenizer, prompt, temperature=1.5):
            new_token = tokens[-1]
            if new_token == "<bos>":
                continue
            
            # Highlight special tokens in different style
            if new_token == "<eos>":
                console.print("[bold magenta]<eos>[/bold magenta]", end="")
            elif new_token in ["<pad>", "<unk>"]:
                console.print(f"[dim]{new_token}[/dim] ", end="")
            else:
                console.print(f"[cyan]{new_token}[/cyan] ", end="")
            
        console.print("\n" + "-"*50 + "\n")



if __name__ == "__main__":
    run_generation_demo()
