from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    refl_legend_loc: str = "upper right"
    sld_legend_loc: str = "upper center"
    sld_inset: bool = False
    sld_inset_bbox: tuple = (0.48, 0.52, 0.42, 0.36)   # x0, y0, width, height in axes coords
    sld_inset_xlim: Optional[tuple] = None
    sld_inset_ylim: Optional[tuple] = None
    sld_inset_show_ticks: bool = True
    sld_inset_linewidth: float = 1.4
    corner_color: str = "#3550a8ff"
    corner_left: float = 0.15
    corner_bottom: float = 0.15
    corner_fontsize: int = 15
    refl_ylabel: str = "Reflectivity (R)"
    refl_xlabel: str = r"Q $[\mathrm{\AA^{-1}}]$"
    sld_ylabel: str = r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$"
    sld_xlabel: str = "Sample depth [Å]"
    marker_styles: Optional[Sequence[str]] = None
    model_linestyles: Optional[Sequence[str]] = None
    model_linewidth: float = 1.8
    data_markersize: float = 3.5
    data_markerfacecolors: Optional[Sequence[str]] = None
    data_markeredgewidth: float = 0.9
    data_markeredgecolors: Optional[Sequence[str]] = None
    axis_linewidth: float = 1.6


def reversed_legend(ax, ncol: Optional[int] = None, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    kwargs = dict(
        loc=loc,
        columnspacing=0.5,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
        fontsize="small",
        handlelength=1.3,
    )
    if ncol is not None:
        kwargs["ncol"] = ncol
    ax.legend(handles[::-1], labels[::-1], **kwargs)


# ---------------------------
# Reflectivity
# ---------------------------

def plot_reflectivity_stage(cfg: StagePlotConfig, save_figs: bool = False, show: bool = True):
    fig, ax = plt.subplots()

    if cfg.marker_styles is None:
        marker_styles = ["o", "s", "o", "s", "o", "s"]
    else:
        marker_styles = list(cfg.marker_styles)

    if cfg.model_linestyles is None:
        model_linestyles = ["-", "-", "-", "-", "-", "-"]
    else:
        model_linestyles = list(cfg.model_linestyles)

    if cfg.data_markerfacecolors is None:
        data_markerfacecolors = ["white"] * len(cfg.colors)
    else:
        data_markerfacecolors = list(cfg.data_markerfacecolors)

    if cfg.data_markeredgecolors is None:
        data_markeredgecolors = list(cfg.colors)
    else:
        data_markeredgecolors = list(cfg.data_markeredgecolors)

    for i, objective in enumerate(cfg.objective_group.objectives):
        y, y_err, model = objective._data_transform(model=objective.generative())

        name = getattr(objective.data, "name", None)
        sigma_safe = np.clip(y_err, 1e-12, None)
        contrib = ((y - model) / sigma_safe) ** 2
        chi2_val = float(np.sum(contrib))
        print(f"{name}: chi2 = {chi2_val:.6f}")

        # data: open markers for stronger grayscale contrast
        ax.errorbar(
            objective.data.x,
            y * cfg.shifts[i],
            y_err * cfg.shifts[i],
            fmt=marker_styles[i],
            ms=cfg.data_markersize,
            mfc=data_markerfacecolors[i],
            mec=data_markeredgecolors[i],
            mew=cfg.data_markeredgewidth,
            linestyle="none",
            color=cfg.colors[i],
            ecolor=cfg.colors[i],
            elinewidth=1,
            capsize=2,
            label=cfg.refl_labels[i],
            zorder=10,
            alpha=1.0,
        )

        # model: same colour family, but line-only and thicker
        ax.plot(
            objective.data.x,
            model * cfg.shifts[i],
            color=cfg.colors[i],
            linestyle=model_linestyles[i],
            lw=cfg.model_linewidth,
            zorder=20,
        )

        ax.set_yscale("log")

        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=None))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_ylabel(cfg.refl_ylabel)
        ax.set_xlabel(cfg.refl_xlabel)
        reversed_legend(ax, ncol=cfg.legend_ncol, loc=cfg.refl_legend_loc)

        for spine in ax.spines.values():
            spine.set_linewidth(cfg.axis_linewidth)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            left=True,
            length=7,
            width=cfg.axis_linewidth * 0.8,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            left=True,
            length=3.5,
            width=cfg.axis_linewidth * 0.5,
        )

        fig.canvas.draw()

    if save_figs:
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.97)
        fig.savefig(cfg.refl_save_path)
        # fig.savefig(cfg.refl_save_path, bbox_inches="tight")

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
    cfg: StagePlotConfig,
    structures: Sequence,
    labels: Sequence[str],
    save_path: str,
    save_figs: bool = False,
    show: bool = True,
    ylabel: str = r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$",
    xlabel: str = "Sample depth [Å]",
    legend_ncol: Optional[int] = None,
    sld_legend_loc: str = "upper middle",
):
    fig, ax = plt.subplots()

    # default: solvent identity by colour, spin identity by linestyle
    sld_linestyles = ["--" if i % 2 == 0 else "-" for i in range(len(structures))]
    sld_colors = list(cfg.colors)

    for i, (structure, label) in enumerate(zip(structures, labels)):
        zed, prof = structure.sld_profile()
        ax.plot(
            zed,
            prof,
            color=sld_colors[i],
            linestyle=sld_linestyles[i],
            linewidth=1.8,
            label=label,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    reversed_legend(ax, ncol=legend_ncol, loc=cfg.sld_legend_loc)

    # match the thicker reflectivity axes
    for spine in ax.spines.values():
        spine.set_linewidth(cfg.axis_linewidth)
    ax.tick_params(which="both", length=7, width=cfg.axis_linewidth * 0.8)

    if cfg.sld_inset:
        x0, y0, w, h = cfg.sld_inset_bbox
        axins = ax.inset_axes([x0, y0, w, h])

        sld_linestyles = ["--" if i % 2 == 0 else "-" for i in range(len(structures))]
        sld_colors = list(cfg.colors)

        for i, (structure, label) in enumerate(zip(structures, labels)):
            zed, prof = structure.sld_profile()
            axins.plot(
                zed,
                prof,
                color=sld_colors[i],
                linestyle=sld_linestyles[i],
                linewidth=cfg.sld_inset_linewidth,
            )

        if cfg.sld_inset_xlim is not None:
            axins.set_xlim(*cfg.sld_inset_xlim)
        if cfg.sld_inset_ylim is not None:
            axins.set_ylim(*cfg.sld_inset_ylim)

        for spine in axins.spines.values():
            spine.set_linewidth(cfg.axis_linewidth * 0.8)

        if cfg.sld_inset_show_ticks:
            axins.tick_params(which="both", direction="in", labelsize=12)
        else:
            axins.set_xticks([])
            axins.set_yticks([])

    if save_figs:
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.97)
        fig.savefig(save_path)
        # fig.savefig(save_path, bbox_inches="tight")

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
        cfg,
        structures=cfg.structures,
        labels=cfg.sld_labels,
        save_path=cfg.sld_save_path,
        save_figs=save_figs,
        show=show,
        ylabel=cfg.sld_ylabel,
        xlabel=cfg.sld_xlabel,
        legend_ncol=cfg.legend_ncol,
        sld_legend_loc=cfg.sld_legend_loc,
    )

    return {
        "refl": (refl_fig, refl_ax),
        "corner": corner_fig,
        "sld": (sld_fig, sld_ax),
    }
