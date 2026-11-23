"""plot_psd: styling options and smoke tests (Agg backend, no display)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from widetrax.Spectra import plot_psd

FREQS = np.linspace(1e-3, 0.5, 128)
PSD = 1e-2 * FREQS ** -3


def _ax():
    fig, ax = plt.subplots()
    return fig, ax


def test_plot_psd_default_no_colors():
    # production variant crashed here: zip(psds, psd_labels, None) -> TypeError
    fig, ax = _ax()
    plot_psd(ax, FREQS, PSD, "cm", "model A", title="t")

    labels = [line.get_label() for line in ax.get_lines()]
    assert "model A" in labels
    assert "$k^{-2}$" in labels
    assert "$k^{-3}$" in labels
    assert "$k^{-5}$" in labels
    assert ax.get_xlabel().startswith("Wavenumber")
    assert ax.get_legend() is not None
    plt.close(fig)


def test_plot_psd_explicit_colors_and_no_legend():
    fig, ax = _ax()
    plot_psd(
        ax, FREQS, [PSD, PSD * 2], "cm", ["A", "B"],
        colors=["tab:red", "tab:blue"], legend=False,
    )
    assert ax.get_legend() is None
    colors = [line.get_color() for line in ax.get_lines()[:2]]
    assert colors == ["tab:red", "tab:blue"]
    assert len(ax.get_lines()) == 5  # 2 PSDs + 3 reference lines
    plt.close(fig)


def test_plot_psd_colors_shorter_than_psds():
    # a short colors list must not drop PSD series
    fig, ax = _ax()
    plot_psd(
        ax, FREQS, [PSD, PSD * 2, PSD * 4], "cm", ["A", "B", "C"],
        colors=["tab:red"],
    )
    assert len(ax.get_lines()) == 6  # 3 PSDs + 3 reference lines
    assert ax.get_lines()[0].get_color() == "tab:red"
    plt.close(fig)


def test_plot_psd_savefig(tmp_path):
    fig, ax = _ax()
    target = tmp_path / "psd.png"
    plot_psd(ax, FREQS, PSD, "cm", "A", savefig=True, filename=str(target))
    assert target.exists() and target.stat().st_size > 0
    plt.close(fig)
