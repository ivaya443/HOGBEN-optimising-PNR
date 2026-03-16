from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import ScalarFormatter


# ---------------------------
# Baseline state utilities
# ---------------------------

def snapshot_params(params: Sequence) -> Dict[object, float]:
    return {p: float(p.value) for p in params}


def restore_params(snapshot: Dict[object, float]) -> None:
    for p, value in snapshot.items():
        p.value = float(value)


def baseline_by_name(snapshot: Dict[object, float]) -> Dict[str, float]:
    return {p.name: float(v) for p, v in snapshot.items()}


def reset_to_fitted_baseline(
    baseline_snapshot: Dict[object, float],
    model_params: Sequence,
    make_structures: Callable[[], list],
):
    restore_params(baseline_snapshot)
    for p in model_params:
        p.optimize = False
    return make_structures()


# ---------------------------
# Context for plot processing
# ---------------------------

@dataclass
class FisherContext:
    MRL: str
    Capping: str
    out_dir: str
    save_figs: bool = True
    eig_powerlimits: tuple = (-3, -3)
    baseline_vals_by_name: Optional[Dict[str, float]] = None
    mrl_display_name: str | None = None
    mrl_thick_name: str = "MRL_thick"
    cap_thick_name: str = "cap_thick"
    mrl_sld_name: str = "MRL_sld"
    mrl_mag_name: str = "MRL_mag"

    @property
    def out_path(self) -> Path:
        return Path(self.out_dir)


# ---------------------------
# Small formatting helpers
# ---------------------------

def _add_baseline_line(ax, param_name: Optional[str], baseline_vals_by_name: Optional[Dict[str, float]]) -> None:
    if not param_name or not baseline_vals_by_name:
        return
    x0 = baseline_vals_by_name.get(param_name)
    if x0 is None:
        return
    ax.axvline(x=x0, color="blue", linestyle="-.", linewidth=1.5, zorder=40)


def _infer_param_name_from_title(title: str, ctx: FisherContext) -> Optional[str]:
    for name in [ctx.mrl_thick_name, ctx.cap_thick_name, ctx.mrl_sld_name, ctx.mrl_mag_name]:
        if name and name in title:
            return name
    return None


def mrl_label(ctx):
    return ctx.mrl_display_name if ctx.mrl_display_name is not None else ctx.MRL

def _set_eigen_x_label(ax, param_name: Optional[str], ctx: FisherContext) -> None:
    if param_name == ctx.mrl_thick_name:
        ax.set_xlabel(rf"Thickness of {mrl_label(ctx)} MRL [Å]")
    elif param_name == ctx.cap_thick_name:
        ax.set_xlabel(r"Thickness of SiO$_2$ Capping Layer [Å]")
    elif param_name == ctx.mrl_sld_name:
        ax.set_xlabel(r"Nuclear SLD of Trial MRL $[\times 10^{-6}\ \mathrm{\AA^{-2}}]$")
    elif param_name == ctx.mrl_mag_name:
        ax.set_xlabel(r"Magnetic SLD of Trial MRL $[\times 10^{-6}\ \mathrm{\AA^{-2}}]$")


# ---------------------------
# Axis processors
# ---------------------------

def process_eigen_axis(ax, ctx: FisherContext, title: str) -> None:
    ax.set_ylabel("Information metric [arb. units]")
    ax.ticklabel_format(style="sci", axis="y", scilimits=ctx.eig_powerlimits)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(ctx.eig_powerlimits)
    ax.yaxis.set_major_formatter(fmt)

    line = None
    for ln in ax.lines:
        xdata = getattr(ln, "get_xdata", lambda: None)()
        ydata = getattr(ln, "get_ydata", lambda: None)()
        if xdata is not None and len(xdata) > 0 and ydata is not None and len(ydata) > 0:
            line = ln
            break

    if line is None:
        return

    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)
    idx_max = int(np.nanargmax(ydata))
    x_at_max = float(xdata[idx_max])
    ax.axvline(x=x_at_max, color="red", linestyle=":", linewidth=1.5, zorder=30)

    param_name = _infer_param_name_from_title(title, ctx)
    _set_eigen_x_label(ax, param_name, ctx)
    _add_baseline_line(ax, param_name, ctx.baseline_vals_by_name)
    ax.set_title("")


def process_sld_axis(ax) -> None:
    ax.set_title("")
    ax.set_ylabel(r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$")
    ax.set_xlabel("Sample depth [Å]")

    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "Structure 0": r"Lipids TRIS D$_2$O ↓",
        "Structure 1": r"Lipids TRIS D$_2$O ↑",
    }
    mapped = [(h, label_map.get(l, l)) for h, l in zip(handles, labels)]
    if len(mapped) >= 2:
        ordered = [mapped[1], mapped[0]]
        new_handles, new_labels = zip(*ordered)
        ax.legend(new_handles, new_labels, loc="upper right", fontsize="small")


def process_reflectivity_axis(ax) -> None:
    ax.set_title("")
    ax.set_ylabel("Reflectivity (R)", fontweight='normal')
    ax.set_xlabel(r"Q $[\mathrm{\AA^{-1}}]$", fontweight='normal')

    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "Model Reflectivity, Structure 0": "Model Reflectivity, Spin ↓",
        "Model Reflectivity, Structure 1": "Model Reflectivity, Spin ↑",
        "Simulated Data, Structure 0": "Simulated Data, Spin ↓",
        "Simulated Data, Structure 1": "Simulated Data, Spin ↑",
    }
    mapped = [(h, label_map.get(l, l)) for h, l in zip(handles, labels)]

    ordered = []
    for prefix in ["Model Reflectivity", "Simulated Data"]:
        ordered += [(h, l) for h, l in mapped if l.startswith(f"{prefix}, Spin ↑")]
        ordered += [(h, l) for h, l in mapped if l.startswith(f"{prefix}, Spin ↓")]

    if ordered:
        new_handles, new_labels = zip(*ordered)
        ax.legend(new_handles, new_labels, loc="upper right", fontsize="small")


def process_hogben_axis(ax, ctx: FisherContext) -> None:
    ylabel = ax.get_ylabel() or ""
    title = ax.get_title() or ""

    if "Minimum eigenvalue" in ylabel:
        process_eigen_axis(ax, ctx, title)
        return

    if "SLD" in ylabel:
        process_sld_axis(ax)
        return

    if "Reflectivity" in ylabel:
        process_reflectivity_axis(ax)
        return

    ax.set_title("")


# ---------------------------
# Figure processing
# ---------------------------

def process_hogben_figures(ctx: FisherContext, filename_stem: str = "Fisher") -> None:
    ctx.out_path.mkdir(parents=True, exist_ok=True)
    fig_nums = plt.get_fignums()

    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        for ax in fig.axes:
            process_hogben_axis(ax, ctx)

        if ctx.save_figs:
            label = ctx.mrl_display_name if ctx.mrl_display_name is not None else ctx.MRL
            out_file = ctx.out_path / f"{filename_stem}_{label}_MRL_{ctx.Capping}_cap_{i}.svg"
            fig.savefig(out_file, format="svg", bbox_inches="tight")
            print(f"Saved: {out_file.name}")
