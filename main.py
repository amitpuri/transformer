import os
import sys
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table

# Add current directory to path so we can import from transformer
sys.path.insert(0, os.path.abspath(os.getcwd()))

from transformer.model import Transformer
from transformer.tokenization import SpacyTokenizer
from transformer import trace

console = Console()

from rich.align import Align

def show_welcome():
    console.print(Align.center(Panel(
        "[bold cyan]Transformer Exploration Toolkit[/bold cyan]\n"
        "[dim]An educational implementation of 'Attention Is All You Need'[/dim]",
        title="Welcome",
        border_style="cyan"
    )))


def run_trace():
    """Runs a step-by-step trace of how text becomes embeddings."""
    from trace_pipeline import run_trace_demo
    run_trace_demo()

def show_architecture():
    """Prints a summary of the Transformer architecture."""
    model = Transformer(
        source_vocab_size=1000,
        target_vocab_size=1000,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=512
    )
    
    table = Table(title="Transformer Architecture (Sample)")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="magenta")
    
    table.add_row("Embedding Dim", "128")
    table.add_row("Encoder Layers", "2")
    table.add_row("Decoder Layers", "2")
    table.add_row("Attention Heads", "4")
    table.add_row("Feed Forward Dim", "512")
    table.add_row("Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    
    console.print(table)
    console.print("\n[dim]Note: This is a scaled-down version for demonstration.[/dim]")

def run_generation():
    """Runs a demo of text generation using greedy decoding."""
    from generation_demo import run_generation_demo
    run_generation_demo()

def main_menu():
    while True:
        status = "[bold green]ON[/bold green]" if trace.is_enabled() else "[bold red]OFF[/bold red]"
        console.print("\n[bold]Main Menu[/bold]")
        console.print("1. [cyan]Trace Embedding Pipeline[/cyan] (Step-by-step visualization)")
        console.print("2. [cyan]View Model Architecture[/cyan] (Summary table)")
        console.print("3. [cyan]Text Generation Demo[/cyan] (Greedy decoding process)")
        console.print(f"4. [cyan]Toggle Deep Trace Mode[/cyan] (Currently: {status})")
        console.print("5. [cyan]Exit[/cyan]")
        
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])
        
        if choice == "1":
            run_trace()
        elif choice == "2":
            show_architecture()
        elif choice == "3":
            run_generation()
        elif choice == "4":
            new_state = trace.toggle()
            console.print(f"[yellow]Deep trace mode is now {'ON' if new_state else 'OFF'}.[/yellow]")
        elif choice == "5":
            console.print("[yellow]Goodbye![/yellow]")
            break


if __name__ == "__main__":
    show_welcome()
    main_menu()
