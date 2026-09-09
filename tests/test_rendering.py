"""The Rich walkthrough and the matplotlib figures the notebook shows."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from tetris import AtomsConfiguration
from tetris.display import format_motions, format_row


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_format_row_marks_the_target_region() -> None:
    row = np.array([True, False, True, False, True, False])
    rendered = format_row(row, target_start=2, target_size=2)
    assert rendered.count("[bold cyan]|[/bold cyan]") == 2
    assert "[green]●[/green]" in rendered
    assert "[yellow]●[/yellow]" in rendered
    assert "[dim]·[/dim]" in rendered


def test_format_motions_skips_atoms_that_stayed() -> None:
    assert format_motions([(3, 3), (5, 5)]) == "[dim]—[/dim]"
    rendered = format_motions([(1, 2), (4, 4)])
    assert "1" in rendered and "2" in rendered
    assert rendered.count("→") == 1


def test_verbose_construction_runs(capsys) -> None:
    config = AtomsConfiguration(
        loading_array_size=10, rng=np.random.default_rng(3)
    )
    config.construct_tetriminoes(verbose=True)
    printed = capsys.readouterr().out
    assert "Loading array size" in printed
    assert "Configuration kept" in printed
    # Rich splits the rule title across style spans, so the
    # label and its index are not one substring
    assert "Row " in printed
    assert "Next shift" in printed


def test_verbose_construction_reports_deficient_columns(capsys) -> None:
    config = AtomsConfiguration(
        occupation_matrix=np.zeros((8, 8), dtype=bool), margin=False
    )
    config.construct_tetriminoes(verbose=True)
    assert "Deficient columns" in capsys.readouterr().out


@pytest.mark.parametrize("which", ["loading", "tetriminoes", "all"])
def test_plot_configuration_produces_a_figure(which: str) -> None:
    config = AtomsConfiguration(
        loading_array_size=14, rng=np.random.default_rng(4)
    )
    figure = config.plot_configuration(which=which)
    expected_axes = 2 if which == "all" else 1
    assert len(figure.axes) == expected_axes
    for ax in figure.axes:
        assert ax.get_title()
    assert config.tetriminoes_matrix is not None or which == "loading"


def test_plot_ticks_stay_readable_for_a_large_array() -> None:
    config = AtomsConfiguration(
        loading_array_size=200, rng=np.random.default_rng(5)
    )
    figure = config.plot_configuration()
    assert len(figure.axes[0].get_xticks()) <= 10
