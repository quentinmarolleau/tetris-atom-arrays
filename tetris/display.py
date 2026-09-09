"""Rich rendering of a tetrimino construction, for the notebook's
step-by-step walkthrough. Nothing here is used by the benchmark."""

import numpy as np
from rich.console import Console
from rich.table import Table

__all__ = [
    "format_motions",
    "format_row",
    "print_header",
    "print_row",
    "print_summary",
]


def format_row(row: np.ndarray, target_start: int, target_size: int) -> str:
    """A boolean row as Rich markup: a bullet per atom, green inside the
    target region and yellow outside to match the plot colours, a dot per
    empty site, and bars delimiting the target region."""
    target_end = target_start + target_size
    symbols = []
    for index, occupied in enumerate(row):
        if not occupied:
            symbols.append("[dim]·[/dim]")
        elif target_start <= index < target_end:
            symbols.append("[green]●[/green]")
        else:
            symbols.append("[yellow]●[/yellow]")
    symbols.insert(target_end, "[bold cyan]|[/bold cyan]")
    symbols.insert(target_start, "[bold cyan]|[/bold cyan]")
    return "  ".join(symbols)


def format_motions(motions: list[tuple[int, int]]) -> str:
    """Source to destination column moves as arrows, skipping the atoms
    that did not actually move."""
    moved = [
        (source, destination)
        for source, destination in motions
        if source != destination
    ]
    if not moved:
        return "[dim]—[/dim]"
    return "  ".join(
        f"[cyan]{source}[/cyan][dim]→[/dim][magenta]{destination}[/magenta]"
        for source, destination in moved
    )


def print_header(
    console: Console,
    loading_array_size: int,
    target_size: int,
    target_start: int,
) -> None:
    info = Table(show_header=False, box=None, pad_edge=False)
    info.add_column(style="bold")
    info.add_row("Loading array size", str(loading_array_size))
    info.add_row("Target size", str(target_size))
    info.add_row("Target start", str(target_start))
    console.print(info)


def print_row(
    console: Console,
    index: int,
    loaded_row: np.ndarray,
    packed_row: np.ndarray,
    motions: list[tuple[int, int]],
    shift: int,
    target_start: int,
    target_size: int,
) -> None:
    moved = sum(
        1 for source, destination in motions if source != destination
    )
    console.rule(f"[bold blue]Row {index}[/bold blue]", style="blue")
    console.print(
        f"[bold]{'Loaded':<10}[/bold] "
        f"{format_row(loaded_row, target_start, target_size)}"
        f"  [dim]({loaded_row.sum()} atoms)[/dim]"
    )
    console.print(
        f"[bold]{'Corrected':<10}[/bold] "
        f"{format_row(packed_row, target_start, target_size)}"
        f"  [dim]({moved} motions)[/dim]"
    )
    console.print(f"[bold]{'Motions':<10}[/bold] {format_motions(motions)}")
    console.print(f"[bold]{'Next shift':<10}[/bold] {shift}")


def print_summary(
    console: Console,
    total_atoms: int,
    total_motions: int,
    configuration_kept: bool,
    deficient_columns: list[tuple[int, int]],
) -> None:
    ratio = total_motions / total_atoms if total_atoms else 0.0
    console.rule(style="blue")
    console.print(f"[bold]Total atoms:[/bold] {total_atoms}")
    console.print(
        f"[bold]Total motions:[/bold] {total_motions}  "
        f"[dim](ratio: {ratio:.3f})[/dim]"
    )
    style = "green" if configuration_kept else "red"
    console.print(
        f"[bold]Configuration kept:[/bold] "
        f"[{style}]{configuration_kept}[/{style}]"
    )
    if not configuration_kept:
        console.print(
            "[bold]Deficient columns:[/bold] "
            + "  ".join(
                f"[red]{column}[/red] [dim]({count} atoms)[/dim]"
                for column, count in deficient_columns
            )
        )
