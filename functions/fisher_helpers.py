from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import ErrorbarContainer
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.patches import Rectangle


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

def _save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    fig.savefig(str(path))
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg
            cairosvg.svg2pdf(
                url=str(path.resolve()),
                write_to=str(path.with_suffix(".pdf")),
            )
        except Exception as e:
            print(f"  PDF conversion skipped ({path.name}): {e}")

# ---------------------------
# Context for plot processing
# ---------------------------

@dataclass
class FisherContext:
    MRL: str
    Capping: str
    outdir: str | None = None
    save_figs: bool = True
    eig_powerlimits: tuple = (-3, -3)
    baseline_vals_by_name: Optional[Dict[str, float]] = None
    mrl_display_name: str | None = None
    mrl_thick_name: str = "MRL_thick"
    cap_thick_name: str = "cap_thick"
    mrl_sld_name: str = "MRL_sld"
    mrl_mag_name: str = "MRL_mag"
    axis_linewidth: float = 1.6
    model_linewidth: float = 1.8
    data_markersize: float = 3.5
    data_markerfacecolors: Optional[Sequence[str]] = None
    data_markeredgewidth: float = 0.9
    data_markeredgecolors: Optional[Sequence[str]] = None
    refl_labels: Optional[Sequence[str]] = None
    sld_labels: Optional[Sequence[str]] = None
    colors: Optional[Sequence[str]] = None
    legend_loc: str = "upper right"
    legend_ncol: int = 2

    structures: Optional[Sequence] = None

    @property
    def out_path(self) -> Path:
        return Path(self.outdir)


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

def _pad_axis_xlim(ax, frac: float = 0.05) -> None:
    xmin, xmax = ax.get_xlim()
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return
    span = xmax - xmin
    if span <= 0:
        return
    pad = frac * span
    ax.set_xlim(xmin - pad, xmax + pad)

def process_eigen_axis(ax, ctx: FisherContext, title: str) -> None:
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits(ctx.eig_powerlimits)
    ax.yaxis.set_major_formatter(fmt)

    # Find the first real data line hogben drew
    line = None
    for ln in ax.lines:
        xd = getattr(ln, "get_xdata", lambda: None)()
        yd = getattr(ln, "get_ydata", lambda: None)()
        if xd is not None and len(xd) > 0 and yd is not None and len(yd) > 0:
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
    _pad_axis_xlim(ax, frac=0.05)
    ax.set_title("")

    # Spine and tick styling to match SLD / reflectivity axes
    for spine in ax.spines.values():
        spine.set_linewidth(ctx.axis_linewidth)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=7, width=ctx.axis_linewidth * 0.8)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=3.5, width=ctx.axis_linewidth * 0.5)

    # Fold the floating exponent label into the ylabel to avoid top-margin clipping
    ax.figure.canvas.draw()
    offset = ax.yaxis.get_offset_text()
    power_str = offset.get_text().strip()
    if power_str:
        offset.set_visible(False)
        ax.set_ylabel(f"Information metric [{power_str} arb. units]")
    else:
        ax.set_ylabel("Information metric [arb. units]")

    # Inset zoom on the peak region for the monotonically decreasing cap_thick plot
    if param_name == ctx.cap_thick_name:
        # --- Domain-informed fixed zoom bounds ---
        zoom_left = -20.0    # small negative headroom left of peak
        zoom_max  = 200.0    # first 200 Å captures the peak and shoulder

        # y bounds driven purely by data in the zoomed x-span
        mask   = (xdata >= zoom_left) & (xdata <= zoom_max)
        y_zoom = ydata[mask]
        if not len(y_zoom):
            return
        y_top     = float(np.nanmax(y_zoom)) * 1.025   # 6% headroom above peak
        y_bot_raw = float(np.nanmin(y_zoom))
        y_bot     = y_bot_raw - 0.08 * (y_top - y_bot_raw)  # symmetric small margin below

        # Overlay rectangle — exact data coords of the zoom region
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (zoom_left, y_bot),
            zoom_max - zoom_left,
            y_top - y_bot,
            linewidth=0.9,
            edgecolor="steelblue",
            facecolor="steelblue",
            alpha=0.12,
            zorder=1,
        )
        ax.add_patch(rect)

        # --- Aspect ratio: inset visual shape == rectangle visual shape ---
        # Both are in axes-fraction space so physical axes dimensions cancel.
        x_lo, x_hi     = ax.get_xlim()
        y_ax_lo, y_ax_hi = ax.get_ylim()
        x_range = x_hi - x_lo
        y_range = y_ax_hi - y_ax_lo

        x_frac = (zoom_max - zoom_left) / x_range   # rectangle width as fraction of x-axis
        y_frac = (y_top   - y_bot)      / y_range   # rectangle height as fraction of y-axis

        w_frac = 0.38                                # target ~1/3 of axis width
        h_frac = w_frac * (y_frac / x_frac)         # preserves visual aspect ratio

        # Pin to top-right with a small margin
        x0_ins = 0.95 - w_frac
        y0_ins = 0.95 - h_frac
        axins = ax.inset_axes([x0_ins, y0_ins, w_frac, h_frac])

        axins.set_zorder(50)
        axins.patch.set_facecolor("white")

        axins.plot(xdata, ydata, color=line.get_color(),
                linewidth=ctx.model_linewidth * 0.85)
        axins.axvline(x=x_at_max, color="red", linestyle=":", linewidth=1.2, zorder=30)
        _add_baseline_line(axins, param_name, ctx.baseline_vals_by_name)

        axins.set_xlim(zoom_left, zoom_max)
        axins.set_ylim(y_bot, y_top)

        # Y ticks: 4-5 clean levels, scientific notation power folded into label
        axins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ins_fmt = ScalarFormatter(useMathText=True)
        ins_fmt.set_powerlimits(ctx.eig_powerlimits)
        axins.yaxis.set_major_formatter(ins_fmt)
        ax.figure.canvas.draw()
        for label in axins.get_xticklabels():
            if label.get_text() == "100":
                label.set_bbox(dict(facecolor="white", edgecolor="none", pad=3))
        axins.yaxis.get_offset_text().set_fontsize(10)

        # X ticks: just a few round-number positions, no decimals
        axins.xaxis.set_major_locator(
            mticker.FixedLocator([0, 50, 100, 150, 200])
        )
        axins.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

        for spine in axins.spines.values():
            spine.set_linewidth(ctx.axis_linewidth * 0.75)
        axins.tick_params(which="major", direction="out", labelsize=9,
                        length=4, width=ctx.axis_linewidth * 0.6)
        axins.tick_params(which="minor", direction="out",
                        length=2, width=ctx.axis_linewidth * 0.4)


def process_sld_axis(ax, ctx: FisherContext) -> None:
    ax.set_title("")
    ax.set_ylabel(r"SLD, $\rho\ [\times 10^{-6}\ \mathrm{\AA^{-2}}]$")
    ax.set_xlabel("Sample depth [Å]")

    # Remove every line hogben drew so we can replace them cleanly.
    for line in ax.lines[:]:
        line.remove()

    if ctx.structures and ctx.colors:
        sld_linestyles = ["--" if i % 2 == 0 else "-" for i in range(len(ctx.structures))]
        for i, structure in enumerate(ctx.structures):
            zed, prof = structure.sld_profile()
            label = (ctx.sld_labels[i] if ctx.sld_labels else f"Structure {i}")
            ax.plot(
                zed, prof,
                color=ctx.colors[i],
                linestyle=sld_linestyles[i],
                linewidth=ctx.model_linewidth,
                label=label,
            )

    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "Structure 0": r"Lipids TRIS D$_2$O ↓",
        "Structure 1": r"Lipids TRIS D$_2$O ↑",
    }
    mapped = [(h, label_map.get(l, l)) for h, l in zip(handles, labels)]
    if len(mapped) >= 2:
        ordered = [mapped[1], mapped[0]]
        new_handles, new_labels = zip(*ordered)
        ax.legend(new_handles, new_labels, loc="upper right", fontsize="small", handlelength=1.3)

    for spine in ax.spines.values():
        spine.set_linewidth(ctx.axis_linewidth)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=7, width=ctx.axis_linewidth * 0.8)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=3.5, width=ctx.axis_linewidth * 0.5)
    _pad_axis_xlim(ax, frac=0.05)
    
    # Suppress negative tick labels (padding artefact) while keeping the visual spacing
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "" if x < 0 else f"{x:g}")
    )

def process_reflectivity_axis(ax, ctx: FisherContext) -> None:
    ax.set_title("")
    ax.set_ylabel("Reflectivity (R)", fontweight="normal")
    ax.set_xlabel(r"Q $[\mathrm{\AA^{-1}}]$", fontweight="normal")

    # Restyle model lines and data markers to match fitting-stage conventions
    # Structure 0 = spin ↓ (solid line, open circle), Structure 1 = spin ↑ (dashed, open square)
    _marker_styles = ["o", "s"]
    _linestyles    = ["-", "--"]

    for line in ax.lines:
        lbl = line.get_label()
        for idx in range(2):
            if f"Model Reflectivity, Structure {idx}" in lbl:
                line.set_linestyle(_linestyles[idx])
                line.set_linewidth(ctx.model_linewidth)

    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        lbl = container.get_label()
        for idx in range(2):
            if f"Simulated Data, Structure {idx}" in lbl:
                data_line, caplines, barlines = container.lines

                data_line.set_marker(_marker_styles[idx])
                data_line.set_markerfacecolor("white")
                data_line.set_markeredgecolor(data_line.get_color())
                data_line.set_markersize(3.8)
                data_line.set_markeredgewidth(1.0)
                data_line.set_alpha(1.0)

                # bar lines — tuple of LineCollection
                for barcol in barlines:
                    barcol.set_linewidth(1)
                    barcol.set_zorder(8)
                    barcol.set_alpha(1.0)

                # cap lines — tuple of Line2D, markersize controls cap length
                for cap in caplines:
                    cap.set_markersize(3.8)
                    cap.set_zorder(8)
                    cap.set_alpha(1.0)
                    
                data_line.set_zorder(10)

    label_map = {
        "Model Reflectivity, Structure 0": "Model Reflectivity, Spin ↓",
        "Model Reflectivity, Structure 1": "Model Reflectivity, Spin ↑",
        "Simulated Data, Structure 0":     "Simulated Data, Spin ↓",
        "Simulated Data, Structure 1":     "Simulated Data, Spin ↑",
    }
    handles, labels = ax.get_legend_handles_labels()
    mapped = [(h, label_map.get(l, l)) for h, l in zip(handles, labels)]

    ordered = []
    for prefix in ["Model Reflectivity", "Simulated Data"]:
        ordered += [(h, l) for h, l in mapped if l.startswith(f"{prefix}, Spin ↑")]
        ordered += [(h, l) for h, l in mapped if l.startswith(f"{prefix}, Spin ↓")]

    if ordered:
        new_handles, new_labels = zip(*ordered)
        ax.legend(new_handles, new_labels, loc="upper right", fontsize="small",
                  columnspacing=0.5, handletextpad=0.4, borderpad=0.3,
                  labelspacing=0.3, handlelength=1.3)

    for spine in ax.spines.values():
        spine.set_linewidth(ctx.axis_linewidth)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=7, width=ctx.axis_linewidth * 0.8)
    ax.tick_params(axis="both", which="minor", direction="out",
                   length=3.5, width=ctx.axis_linewidth * 0.5)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

def process_hogben_axis(ax, ctx: FisherContext) -> None:
    ylabel = ax.get_ylabel() or ""
    title = ax.get_title() or ""

    if "Minimum eigenvalue" in ylabel:
        process_eigen_axis(ax, ctx, title)
        return

    if "SLD" in ylabel:
        process_sld_axis(ax, ctx)
        return

    if "Reflectivity" in ylabel:
        process_reflectivity_axis(ax, ctx)
        return

    ax.set_title("")


# ---------------------------
# Figure processing
# ---------------------------

def _classify_figure(fig, ctx: FisherContext) -> str:
    for ax in fig.axes:
        ylabel = ax.get_ylabel() or ""
        title  = ax.get_title()  or ""
        if "Minimum eigenvalue" in ylabel:
            print(f"  DEBUG eigen title: {repr(title)}")  # <-- add this
            print(f"  DEBUG looking for: {repr(ctx.mrl_thick_name)}, {repr(ctx.cap_thick_name)}")

            param = _infer_param_name_from_title(title, ctx)
            if param == ctx.mrl_thick_name:
                return f"eigen_{ctx.MRL}_thick"
            if param == ctx.cap_thick_name:
                return f"eigen_{ctx.Capping}_thick"
            if param == ctx.mrl_sld_name:
                return f"eigen_{ctx.MRL}_nsld"
            if param == ctx.mrl_mag_name:
                return f"eigen_{ctx.MRL}_msld"
            return "eigen_unknown"
        if "SLD" in ylabel:
            return "sld"
        if "Reflectivity" in ylabel:
            return "refl"
    return "unknown"


def process_hogben_figures(ctx: FisherContext, filename_stem: str = "Fisher") -> dict[str, Path]:
    ctx.out_path.mkdir(parents=True, exist_ok=True)
    label = ctx.mrl_display_name if ctx.mrl_display_name is not None else ctx.MRL
    saved: dict[str, Path] = {}

    for num in plt.get_fignums():
        fig = plt.figure(num)

        # Classify BEFORE processing clears titles
        key = _classify_figure(fig, ctx)

        for ax in fig.axes:
            process_hogben_axis(ax, ctx)

        # Always register the key, regardless of whether saving succeeds
        out_file = ctx.out_path / f"{filename_stem}_{label}_MRL_{ctx.Capping}_cap_{key}.svg"
        saved[key] = out_file

        if ctx.save_figs:
            fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.97)
            try:
                _save_figure(fig, out_file)
                print(f"Saved: {out_file.name}")
            except Exception as e:
                print(f"Warning: could not save {out_file.name}: {e}")
                # Still keep the key so assemble_panel can report a cleaner error

    return saved


def assemble_panel(
    svg_paths: Sequence[str | Path],
    ncols: int,
    out_path: str | Path,
    labels: Optional[Sequence[str]] = None,
    gap: float = 24.0,
    label_fontsize: int = 18,
    label_x_offset: float = 12.0,
    label_y_offset: float = 20.0,  # up from bottom-left corner of each figure
) -> None:
    import xml.etree.ElementTree as ET
    import base64

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svgns = "http://www.w3.org/2000/svg"

    # Read actual figure size from first file
    first_root = ET.parse(str(svg_paths[0])).getroot()
    fig_w = fig_h = None
    for elem in first_root.iter():
        try:
            fig_w = float(elem.get("width", ""))
            fig_h = float(elem.get("height", ""))
            if fig_w > 0 and fig_h > 0:
                break
        except (ValueError, TypeError):
            continue
    if fig_w is None:
        raise ValueError(f"Could not read dimensions from {svg_paths[0]}")

    n = len(svg_paths)
    nrows = (n + ncols - 1) // ncols
    cell_w = fig_w + gap
    cell_h = fig_h + gap

    if labels is None:
            labels = [chr(97 + i) for i in range(n)]  # a, b, c ... (parens added in markup)
    
    panel_w = ncols * cell_w - gap
    panel_h = nrows * cell_h - gap

    panel = ET.Element(f"{{{svgns}}}svg", {
        "width":   str(panel_w),
        "height":  str(panel_h),
        "viewBox": f"0 0 {panel_w} {panel_h}",
        "version": "1.1",
    })

    offsets = []

    # Pass 1: embed each figure as a base64 SVG image -- no tree surgery, no ID conflicts
    for i, path in enumerate(svg_paths):
        row, col = divmod(i, ncols)
        x_off = col * cell_w
        y_off = row * cell_h
        offsets.append((x_off, y_off))

        with open(str(path), "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        ET.SubElement(panel, f"{{{svgns}}}image", {
            "x":      str(x_off),
            "y":      str(y_off),
            "width":  str(fig_w),
            "height": str(fig_h),
            "href":   f"data:image/svg+xml;base64,{encoded}",
        })

    # Pass 2: labels last -- always on top
    for i, (x_off, y_off) in enumerate(offsets):
        lbl = ET.SubElement(panel, f"{{{svgns}}}text", {
            "x":           str(x_off + label_x_offset),
            "y":           str(y_off + fig_h - label_y_offset),
            "font-size":   str(label_fontsize),
            "font-family": "sans-serif",
            "fill":        "black",
        })
        letter = labels[i] if labels is not None else chr(97 + i)
        lbl.text = "("
        tspan = ET.SubElement(lbl, f"{{{svgns}}}tspan", {"font-style": "italic"})
        tspan.text = letter
        tspan.tail = ")"

    tree = ET.ElementTree(panel)
    tree.write(str(out_path), xml_declaration=True, encoding="unicode")
    print(f"Panel saved: {Path(out_path).name} "
          f"({ncols}×{nrows}, cell {fig_w:.0f}×{fig_h:.0f} pt + {gap:.0f} pt gap)")
    