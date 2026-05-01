"""
Trace utility for the Transformer walkthrough.

When enabled, every component prints what it's doing as data flows through,
showing tensor shapes, key computations, and the logic at each step.
"""
import torch
from rich.console import Console

console = Console()

_enabled = False
_depth = 0


def enable():
    """Turn on tracing globally."""
    global _enabled
    _enabled = True


def disable():
    """Turn off tracing globally."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def toggle() -> bool:
    """Toggle tracing on/off. Returns the new state."""
    global _enabled
    _enabled = not _enabled
    return _enabled


def _indent() -> str:
    return "|  " * _depth


def enter(component: str, detail: str = ""):
    """Mark entry into a component."""
    global _depth
    if not _enabled:
        return
    pad = "|  " * _depth
    msg = f"[bold cyan]{component}[/bold cyan]"
    if detail:
        msg += f"  [dim]{detail}[/dim]"
    console.print(f"{pad}+- {msg}")
    _depth += 1


def exit(component: str = "", summary: str = ""):
    """Mark exit from a component."""
    global _depth
    if not _enabled:
        return
    _depth = max(0, _depth - 1)
    pad = "|  " * _depth
    msg = ""
    if summary:
        msg = f"  [green]{summary}[/green]"
    console.print(f"{pad}+-{msg}")


def log(message: str, style: str = ""):
    """Log a trace message at the current depth."""
    if not _enabled:
        return
    pad = "|  " * _depth
    if style:
        console.print(f"{pad}[{style}]{message}[/{style}]")
    else:
        console.print(f"{pad}{message}")


def tensor(name: str, t: torch.Tensor, show_stats: bool = False):
    """Log a tensor's shape and optionally its stats."""
    if not _enabled:
        return
    pad = "|  " * _depth
    info = f"[*] [yellow]{name}[/yellow]: shape={tuple(t.shape)}"
    if show_stats and t.numel() > 0:
        info += f"  mean={t.float().mean().item():.4f}  std={t.float().std().item():.4f}"
    console.print(f"{pad}{info}")


def divider(label: str = ""):
    """Print a visual divider."""
    if not _enabled:
        return
    pad = "|  " * _depth
    if label:
        console.print(f"{pad}[dim]-- {label} --[/dim]")
    else:
        console.print(f"{pad}[dim]----------[/dim]")
