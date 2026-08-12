"""
Figures for the three-way reversal study.

Primary figure: reversal-aligned recovery curves with bootstrap confidence
bands, baseline / emotional / yoked overlaid. Secondary: mood aligned to
reversal onset (carrying the PASS/FAIL gate), and path-type composition over
the run.

Usage:
    python visualize_reversal.py --study_dir experiments/reversal_study_main
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.reversal_analysis import (
    PRIMARY_METRIC,
    align_blocks,
    analyze_study,
    bootstrap_ci,
    discover_runs,
    load_schedule,
    mood_aligned_to_reversal,
    path_type_over_time,
    print_report,
)

# Validated categorical slots 1-3 (light surface #fcfcfb): blue / orange / aqua.
# Fixed assignment by agent identity, never by rank, so a figure that drops an
# arm does not repaint the survivors.
SERIES_COLOR = {
    "baseline": "#2a78d6",
    "emotional": "#eb6834",
    "yoked": "#1baf7a",
}
SERIES_ORDER = ["baseline", "emotional", "yoked"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e2"


def _style_axes(ax) -> None:
    """Recessive grid and axes; the data is the only prominent thing."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _run_curves(
    study_dir: str,
    window: int,
    exclude_first_reversal: bool,
    only_acquirers: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """Per-agent list of run-level reversal-aligned adherence curves."""
    curves: Dict[str, List[np.ndarray]] = {}
    for run in discover_runs(study_dir):
        if only_acquirers and not run["reached_criterion"]:
            continue
        rows = load_schedule(run["schedule_csv"])
        aligned = align_blocks(rows, window, exclude_first_reversal)
        if not aligned.get("n_blocks"):
            continue
        curves.setdefault(run["agent_type"], []).append(
            aligned["mean_adherence_smoothed"]
        )
    return curves


def plot_recovery_curves(
    study_dir: str,
    output_path: str,
    window: int = 20,
    exclude_first_reversal: bool = True,
    n_boot: int = 2000,
) -> Optional[str]:
    """THE primary figure: adherence to the now-optimal route after a reversal.

    Each run contributes one curve (its own mean across reversals 2..R), so
    runs are the unit of analysis and a long run cannot dominate. Bands are
    percentile bootstrap CIs over runs.
    """
    curves = _run_curves(study_dir, window, exclude_first_reversal)
    if not curves:
        print("No completed runs with reversal blocks yet — skipping recovery figure")
        return None

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _style_axes(ax)

    length = min(min(len(c) for c in runs) for runs in curves.values())
    x = np.arange(1, length + 1)

    for agent_type in SERIES_ORDER:
        runs = curves.get(agent_type)
        if not runs:
            continue
        stack = np.stack([c[:length] for c in runs])
        # Columns before the rolling window fills are all-NaN by construction;
        # averaging them warns and yields NaN, which we plot as a gap.
        with np.errstate(invalid="ignore"):
            mean = np.where(
                np.isnan(stack).all(axis=0), np.nan,
                np.nanmean(np.where(np.isnan(stack), np.nan, stack), axis=0),
            )
        lo, hi = np.full(length, np.nan), np.full(length, np.nan)
        if len(stack) >= 2:
            for i in range(length):
                col = stack[:, i]
                lo[i], hi[i] = bootstrap_ci(col, n_boot=n_boot, seed=i)

        color = SERIES_COLOR[agent_type]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.0, zorder=3,
                label=f"{agent_type} (n={len(runs)})")

        # Direct label at the curve end — identity is never color-alone, and
        # this is the relief the aqua slot's contrast WARN requires.
        last = np.where(~np.isnan(mean))[0]
        if len(last):
            i = last[-1]
            ax.annotate(agent_type, (x[i], mean[i]), xytext=(6, 0),
                        textcoords="offset points", color=color,
                        fontsize=9, fontweight="medium", va="center")

    ax.axvline(window, color=INK_SECONDARY, linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate(f"rolling window ({window}) filled",
                (window, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=INK_SECONDARY, fontsize=8)

    ax.set_xlabel("Episodes since reversal", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"Adherence to now-optimal route\n(rolling {window} episodes)",
                  color=INK_SECONDARY, fontsize=10)
    ax.set_title("Recovery after contingency reversal",
                 color=INK, fontsize=13, fontweight="semibold", loc="left", pad=14)
    ax.text(0, 1.02, f"Mean across reversals 2..R per run, then across runs; "
                     f"bands are 95% bootstrap CIs over runs",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, length + max(6, length * 0.06))
    leg = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {output_path}")
    return output_path


def plot_mood_alignment(
    study_dir: str,
    output_path: str,
    pre: int = 20,
    post: int = 40,
    exclude_first_reversal: bool = True,
) -> Optional[str]:
    """M(t) aligned to reversal onset, for every mood-carrying agent type.

    If these curves are flat through the onset line, the mood mechanism is not
    engaging and the recovery figure cannot be attributed to it.
    """
    per_agent: Dict[str, List[np.ndarray]] = {}
    for run in discover_runs(study_dir):
        if run["agent_type"] not in ("emotional", "yoked"):
            continue
        rows = load_schedule(run["schedule_csv"])
        aligned = mood_aligned_to_reversal(rows, pre, post, exclude_first_reversal)
        if aligned is None or aligned.size == 0:
            continue
        per_agent.setdefault(run["agent_type"], []).append(np.nanmean(aligned, axis=0))

    if not per_agent:
        print("No mood-carrying runs yet — skipping mood figure")
        return None

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style_axes(ax)
    x = np.arange(-pre, post)

    for agent_type in SERIES_ORDER:
        runs = per_agent.get(agent_type)
        if not runs:
            continue
        stack = np.stack(runs)
        mean = np.nanmean(stack, axis=0)
        lo, hi = np.full(len(x), np.nan), np.full(len(x), np.nan)
        if len(stack) >= 2:
            for i in range(len(x)):
                lo[i], hi[i] = bootstrap_ci(stack[:, i], n_boot=2000, seed=i)
        color = SERIES_COLOR[agent_type]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.0, zorder=3,
                label=f"{agent_type} (n={len(runs)})")

    ax.axvline(0, color=INK, linewidth=1.2, zorder=4)
    ax.annotate("reversal", (0, ax.get_ylim()[1]), xytext=(5, -12),
                textcoords="offset points", color=INK, fontsize=9)
    ax.axhline(0, color=GRID, linewidth=1.0, zorder=1)

    ax.set_xlabel("Episodes relative to reversal onset", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Mean mood M", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Does mood respond to the contingency change?",
                 color=INK, fontsize=13, fontweight="semibold", loc="left", pad=14)
    ax.text(0, 1.02, "A flat line through the onset means the mechanism is not engaging",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)
    leg = ax.legend(frameon=False, loc="best", fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {output_path}")
    return output_path


def plot_path_type_composition(
    study_dir: str,
    output_path: str,
    bin_size: int = 50,
) -> Optional[str]:
    """Path-type share over time, one panel per agent type (small multiples)."""
    runs_by_agent: Dict[str, List[List[Dict[str, Any]]]] = {}
    for run in discover_runs(study_dir):
        runs_by_agent.setdefault(run["agent_type"], []).append(
            load_schedule(run["schedule_csv"])
        )
    if not runs_by_agent:
        return None

    agents = [a for a in SERIES_ORDER if a in runs_by_agent]
    fig, axes = plt.subplots(len(agents), 1, figsize=(9, 2.4 * len(agents)),
                             facecolor=SURFACE, sharex=True, squeeze=False)

    # Sequential-by-identity: each path type keeps one hue across all panels.
    type_color = {
        "shield_route": "#2a78d6",
        "trap_rush": "#eb6834",
        "timeout": "#52514e",
        "direct": "#1baf7a",
        "key_route": "#eda100",
        "other": "#c3c2b7",
    }

    for ax, agent_type in zip(axes[:, 0], agents):
        _style_axes(ax)
        runs = runs_by_agent[agent_type]
        n_bins = min(len(path_type_over_time(r, bin_size)[
            next(iter(path_type_over_time(r, bin_size)))]) for r in runs)
        types = sorted({t for r in runs for t in path_type_over_time(r, bin_size)})
        shares = {
            t: np.mean([
                np.array(path_type_over_time(r, bin_size).get(t, [0] * n_bins)[:n_bins])
                for r in runs
            ], axis=0)
            for t in types
        }
        x = np.arange(n_bins) * bin_size
        bottom = np.zeros(n_bins)
        for t in types:
            ax.fill_between(x, bottom, bottom + shares[t], step="post",
                            color=type_color.get(t, "#c3c2b7"), alpha=0.9,
                            linewidth=0, label=t)
            bottom = bottom + shares[t]
        ax.set_ylim(0, 1)
        ax.set_ylabel(agent_type, color=INK_SECONDARY, fontsize=10)

    axes[-1, 0].set_xlabel(f"Episode (bins of {bin_size})",
                           color=INK_SECONDARY, fontsize=10)
    axes[0, 0].set_title("Path-type composition over training",
                         color=INK, fontsize=13, fontweight="semibold",
                         loc="left", pad=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, ncol=len(labels),
                     loc="lower center", fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Reversal study figures")
    parser.add_argument("--study_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--recovery_fraction", type=float, default=0.80)
    parser.add_argument("--include_first_reversal", action="store_true",
                        help="Include reversal 1, which is confounded with the "
                             "end of acquisition (the pre-registration excludes it)")
    parser.add_argument("--bin_size", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or Path(args.study_dir) / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude_first = not args.include_first_reversal

    analysis = analyze_study(
        args.study_dir,
        window=args.window,
        recovery_fraction=args.recovery_fraction,
        exclude_first_reversal=exclude_first,
    )
    print_report(analysis)
    with open(out_dir / "reversal_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  wrote {out_dir / 'reversal_analysis.json'}")

    plot_recovery_curves(args.study_dir, str(out_dir / "recovery_curves.png"),
                         window=args.window, exclude_first_reversal=exclude_first)
    plot_mood_alignment(args.study_dir, str(out_dir / "mood_alignment.png"),
                        pre=args.window, post=2 * args.window,
                        exclude_first_reversal=exclude_first)
    plot_path_type_composition(args.study_dir,
                               str(out_dir / "path_type_composition.png"),
                               bin_size=args.bin_size)


if __name__ == "__main__":
    main()
