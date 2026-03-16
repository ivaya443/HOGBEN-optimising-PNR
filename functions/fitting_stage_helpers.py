from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StagePlotConfig:
    objective_group: object
    structures: Sequence
    refl_labels: Sequence[str]
    sld_labels: Sequence[str]
    colors: Sequence[str]
    shifts: Sequence[float]
    refl_save_path: str
    sld_save_path: str
    corner_save_path: Optional[str] = None
    legend_ncol: Optional[int] = None
    legend_loc: str = "upper right"
    corner_color: str = "#3550a8ff"
    corner_left: float = 0.15
    corner_bottom: float = 0.15
    corner_fontsize: int = 15
    refl_ylabel: str = "Reflectivity (R)"
    refl_xlabel: str = r"Q $[\mathrm{\AA^{-1}}]$"
    sld_ylabel: str = r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$"
    sld_xlabel: str = "Sample depth [Å]"


def reversed_legend(ax, ncol: Optional[int] = None, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    kwargs = dict(
        loc=loc,
        columnspacing=0.5,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.3,
        fontsize="small",
        handlelength=1,
    )
    if ncol is not None:
        kwargs["ncol"] = ncol
    ax.legend(handles[::-1], labels[::-1], **kwargs)


# ---------------------------
# Reflectivity
# ---------------------------

def plot_reflectivity_stage(cfg: StagePlotConfig, save_figs: bool = False, show: bool = True):
    fig, ax = plt.subplots()

    for i, objective in enumerate(cfg.objective_group.objectives):
        y, y_err, model = objective._data_transform(model=objective.generative())

        name = getattr(objective.data, "name", None)
        sigma_safe = np.clip(y_err, 1e-12, None)
        contrib = ((y - model) / sigma_safe) ** 2
        chi2_val = float(np.sum(contrib))
        print(f"{name}: chi2 = {chi2_val:.6f}")

        ax.errorbar(
            objective.data.x,
            y * cfg.shifts[i],
            y_err * cfg.shifts[i],
            fmt="o",
            ms=3,
            color=cfg.colors[i],
            ecolor=cfg.colors[i],
            elinewidth=1,
            capsize=2,
            label=cfg.refl_labels[i],
            zorder=10,
            alpha=0.9,
        )

        ax.plot(
            objective.data.x,
            model * cfg.shifts[i],
            color=cfg.colors[i],
            lw=1.5,
            zorder=20,
        )

    ax.set_yscale("log")
    ax.set_ylabel(cfg.refl_ylabel)
    ax.set_xlabel(cfg.refl_xlabel)
    reversed_legend(ax, ncol=cfg.legend_ncol, loc=cfg.legend_loc)

    if save_figs:
        fig.savefig(cfg.refl_save_path, bbox_inches="tight")

    if show:
        plt.show()
    return fig, ax


# ---------------------------
# Corner
# ---------------------------

def restyle_corner_labels(fig, left: float = 0.15, bottom: float = 0.15, fontsize: int = 15):
    axes = fig.axes
    n = int(len(axes) ** 0.5)

    for i in range(n):
        for j in range(n):
            ax = axes[i * n + j]
            if i == n - 1:
                label = ax.xaxis.get_label()
                pos = label.get_position()
                offset = -0.4 if j % 2 == 0 else -0.6
                label.set_position((pos[0], offset))
            if j == 0:
                label = ax.yaxis.get_label()
                pos = label.get_position()
                offset = -0.4 if i % 2 == 0 else -0.6
                label.set_position((offset, pos[1]))

    fig.subplots_adjust(left=left, bottom=bottom)
    for text in fig.findobj(match=lambda x: hasattr(x, "set_fontsize")):
        text.set_fontsize(fontsize)



def sample_and_plot_corner(
    fitter,
    objective_group,
    save_path: Optional[str],
    do_sampling: bool,
    save_figs: bool,
    show: bool = True,
    color: str = "#3550a8ff",
    left: float = 0.15,
    bottom: float = 0.15,
    fontsize: int = 15,
):
    if not do_sampling:
        return None

    fitter.sample(400, random_state=1)
    fitter.sampler.reset()
    fitter.sample(20, nthin=50, random_state=1)

    corner_fig = objective_group.corner(
        show_titles=False,
        fill_contours=True,
        color=color,
        hist_kwargs={
            "color": color,
            "alpha": 1,
            "edgecolor": color,
        },
    )

    restyle_corner_labels(corner_fig, left=left, bottom=bottom, fontsize=fontsize)

    if save_figs and save_path is not None:
        corner_fig.savefig(save_path)

    if show:
        plt.show()
    return corner_fig


# ---------------------------
# SLD
# ---------------------------

def plot_sld_stage(
    structures: Sequence,
    labels: Sequence[str],
    save_path: str,
    save_figs: bool = False,
    show: bool = True,
    ylabel: str = r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$",
    xlabel: str = "Sample depth [Å]",
    legend_ncol: Optional[int] = None,
    legend_loc: str = "upper right",
):
    fig, ax = plt.subplots()

    for structure, label in zip(structures, labels):
        ax.plot(*structure.sld_profile(), label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    reversed_legend(ax, ncol=legend_ncol, loc=legend_loc)

    if save_figs:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    return fig, ax


# ---------------------------
# High-level wrapper
# ---------------------------

def run_fitting_stage_plots(
    cfg: StagePlotConfig,
    fitter=None,
    do_sampling: bool = False,
    save_figs: bool = False,
    show: bool = True,
):
    refl_fig, refl_ax = plot_reflectivity_stage(cfg, save_figs=save_figs, show=show)

    corner_fig = sample_and_plot_corner(
        fitter=fitter,
        objective_group=cfg.objective_group,
        save_path=cfg.corner_save_path,
        do_sampling=do_sampling,
        save_figs=save_figs,
        show=show,
        color=cfg.corner_color,
        left=cfg.corner_left,
        bottom=cfg.corner_bottom,
        fontsize=cfg.corner_fontsize,
    )

    sld_fig, sld_ax = plot_sld_stage(
        structures=cfg.structures,
        labels=cfg.sld_labels,
        save_path=cfg.sld_save_path,
        save_figs=save_figs,
        show=show,
        ylabel=cfg.sld_ylabel,
        xlabel=cfg.sld_xlabel,
        legend_ncol=cfg.legend_ncol,
        legend_loc=cfg.legend_loc,
    )

    return {
        "refl": (refl_fig, refl_ax),
        "corner": corner_fig,
        "sld": (sld_fig, sld_ax),
    }
