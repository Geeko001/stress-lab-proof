"""Matplotlib helpers — return figures, never show directly."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def new_fig(xlabel="", ylabel="", title=""):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_accuracy_vs_bindings(m_vals, linear_acc, softmax_acc, title="Retrieval accuracy vs bindings"):
    fig, ax = new_fig("Number of stored bindings (m)", "Retrieval accuracy", title)
    ax.set_xscale("log", base=2)
    ax.plot(m_vals, linear_acc, marker="o", label="Linear attention")
    ax.plot(m_vals, softmax_acc, marker="s", label="Softmax reference")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_multi_curve(x_vals, curves, xlabel="", ylabel="", title="", xscale="linear"):
    fig, ax = new_fig(xlabel, ylabel, title)
    if xscale == "log":
        ax.set_xscale("log", base=2)
    for label, y in curves.items():
        ax.plot(x_vals, y, marker="o", label=str(label))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_simple(x, y, xlabel="", ylabel="", title="", marker="o"):
    fig, ax = new_fig(xlabel, ylabel, title)
    ax.plot(x, y, marker=marker)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def apply_dark_mode(fig):
    """Restyle a figure for dark video backgrounds. Mutates in place."""
    fig.patch.set_facecolor("#0E1117")
    for ax in fig.axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, alpha=0.2)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor("#0E1117")
            leg.get_frame().set_edgecolor("white")
            for text in leg.get_texts():
                text.set_color("white")
    fig.tight_layout()
    return fig


def save_high_res(fig, path, dpi=300, dark_mode=False):
    """Save a high-resolution PNG for video overlays.

    Creates parent directories as needed. Returns the path as a string.
    """
    if dark_mode:
        apply_dark_mode(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)
