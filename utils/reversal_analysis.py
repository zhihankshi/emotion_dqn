"""
Reversal-aligned analysis for the three-way study.

Everything here slices episode logs into windows aligned to **reversal onset**,
averages within a run across reversals, then across runs — so the unit of
analysis is the run, not the episode, and a run with more reversals cannot
outvote one with fewer.

Pre-registered primary metric
-----------------------------
``mean episodes-to-recovery, averaged across reversals 2..R``.

Reversal 1 is excluded by design: it is confounded with the end of
acquisition (the agent has just left a criterion-terminated phase, its buffer
is full of acquisition data, and its epsilon has only just settled). Fix this
before looking at results — it is a pre-registration, not a tuning knob.

Recovery is defined on **adherence to the currently-optimal route**, not raw
return: the two contingencies have very different optimal returns, so a
return-based "80% of pre-flip level" would compare incomparable quantities.

Mood diagnostic
---------------
``mood_dip_test`` asks the question everything downstream depends on: does M
measurably drop after a reversal? If it does not, the mood mechanism is not
engaging and no comparison of adaptation speed can be attributed to it. This
is surfaced as an explicit PASS/FAIL, not just a plot.
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Path type that is optimal under each contingency.
OPTIMAL_PATH_TYPE = {"protective": "shield_route", "non_protective": "trap_rush"}

# Agents that carry a mood term; the baseline has none, so the mood gate does
# not apply to it.
MOOD_CARRYING_AGENTS = ("emotional", "yoked")

PRIMARY_METRIC = "mean_episodes_to_recovery_reversals_2_to_R"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_schedule(path: str) -> List[Dict[str, Any]]:
    """Load a reversal_schedule.csv, coercing numeric columns."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episode": int(row["episode"]),
                "phase": row["phase"],
                "block": int(row["block"]),
                "contingency": row["contingency"],
                "reversal_index": int(row["reversal_index"]),
                "epsilon": float(row["epsilon"]),
                "path_type": row["path_type"],
                "success": int(row["success"]),
                "total_reward": float(row["total_reward"]),
                "steps": int(row["steps"]),
                "mean_mood": float(row["mean_mood"]),
            })
    return rows


def load_mood_trace(path: str) -> List[Tuple[int, int, float]]:
    """Load mood_trace.csv as (step, episode, mood) rows."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((int(row["step"]), int(row["episode"]), float(row["mood"])))
    return out


def discover_runs(study_dir: str) -> List[Dict[str, Any]]:
    """Find every completed cell of a study directory."""
    study_dir = Path(study_dir)
    runs = []
    for manifest_path in sorted(study_dir.glob("*/*/*/reversal_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent
        runs.append({
            "agent_type": manifest["agent_type"],
            "seed": manifest["seed"],
            "reached_criterion": manifest["reached_criterion"],
            "acquisition_episodes": manifest["acquisition_episodes"],
            "reversals": manifest["reversals"],
            "reversal_period": manifest["reversal_period"],
            "schedule_csv": str(run_dir / "reversal_schedule.csv"),
            "mood_trace_csv": (
                str(run_dir / "mood_trace.csv")
                if (run_dir / "mood_trace.csv").exists() else None
            ),
            "run_dir": str(run_dir),
        })
    return runs


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def split_blocks(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Episodes grouped by reversal block, in order (block 1..R)."""
    blocks: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        if row["phase"] != "reversal":
            continue
        blocks.setdefault(row["block"], []).append(row)
    return [blocks[b] for b in sorted(blocks)]


def optimal_adherence(block: Sequence[Dict[str, Any]]) -> np.ndarray:
    """1/0 per episode: did it take the route that is optimal *now*?"""
    target = OPTIMAL_PATH_TYPE[block[0]["contingency"]]
    return np.array([float(r["path_type"] == target) for r in block])


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean; entries before the window is full are NaN."""
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        c = np.cumsum(np.insert(values, 0, 0.0))
        out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def align_blocks(
    rows: Sequence[Dict[str, Any]],
    window: int = 20,
    exclude_first_reversal: bool = True,
) -> Dict[str, Any]:
    """Per-run, reversal-aligned adherence and return curves.

    Returns arrays of shape (n_blocks, block_length) plus the within-run mean
    across blocks — the run-level curve that goes into the group average.
    """
    blocks = split_blocks(rows)
    if exclude_first_reversal:
        blocks = blocks[1:]
    if not blocks:
        return {"n_blocks": 0}

    length = min(len(b) for b in blocks)
    adherence = np.stack([optimal_adherence(b)[:length] for b in blocks])
    returns = np.stack([[r["total_reward"] for r in b][:length] for b in blocks])
    smoothed = np.stack([rolling_mean(a, window) for a in adherence])

    return {
        "n_blocks": len(blocks),
        "block_length": length,
        "adherence": adherence,
        "adherence_smoothed": smoothed,
        "returns": returns,
        "mean_adherence": adherence.mean(axis=0),
        "mean_adherence_smoothed": np.nanmean(smoothed, axis=0),
        "mean_returns": returns.mean(axis=0),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def pre_flip_level(
    rows: Sequence[Dict[str, Any]],
    block_index: int,
    window: int = 20,
) -> float:
    """Optimal-route adherence in the window *before* a given reversal block.

    For block 1 that is the tail of acquisition; for later blocks it is the
    tail of the previous block, scored against that block's own optimum.
    """
    blocks = split_blocks(rows)
    if block_index == 0:
        acq = [r for r in rows if r["phase"] == "acquisition"][-window:]
        if not acq:
            return 0.0
        target = OPTIMAL_PATH_TYPE["protective"]
        return float(np.mean([r["path_type"] == target for r in acq]))
    prev = blocks[block_index - 1][-window:]
    return float(np.mean(optimal_adherence(prev)))


def episodes_to_recovery(
    rows: Sequence[Dict[str, Any]],
    window: int = 20,
    recovery_fraction: float = 0.80,
    exclude_first_reversal: bool = True,
) -> Dict[str, Any]:
    """Episodes-to-recovery per reversal, and the pre-registered mean.

    A block where adherence never reaches the threshold is right-censored at
    the block length and counted, never dropped: dropping them would bias the
    mean toward whichever agent happens to recover at all.
    """
    blocks = split_blocks(rows)
    per_block: List[Dict[str, Any]] = []

    for i, block in enumerate(blocks):
        baseline_level = pre_flip_level(rows, i, window)
        threshold = recovery_fraction * baseline_level
        series = rolling_mean(optimal_adherence(block), window)

        recovery = None
        if baseline_level > 0:
            for j, value in enumerate(series):
                if not np.isnan(value) and value >= threshold:
                    recovery = j + 1
                    break

        per_block.append({
            "reversal_index": i + 1,
            "contingency": block[0]["contingency"],
            "pre_level": round(baseline_level, 4),
            "threshold": round(threshold, 4),
            "recovery_episode": recovery,
            "censored": baseline_level > 0 and recovery is None,
            "valid": baseline_level > 0,
            "block_length": len(block),
        })

    scored = per_block[1:] if exclude_first_reversal else per_block
    usable = [b for b in scored if b["valid"]]
    filled = [
        b["recovery_episode"] if b["recovery_episode"] is not None else b["block_length"]
        for b in usable
    ]

    return {
        "per_block": per_block,
        "n_scored_blocks": len(usable),
        "n_censored": sum(b["censored"] for b in usable),
        PRIMARY_METRIC: float(np.mean(filled)) if filled else None,
        "excluded_first_reversal": exclude_first_reversal,
    }


def perseveration(
    rows: Sequence[Dict[str, Any]],
    exclude_first_reversal: bool = True,
) -> Dict[str, Any]:
    """Episodes still taking the old-optimal route before the first new-optimal one."""
    blocks = split_blocks(rows)
    scored = blocks[1:] if exclude_first_reversal else blocks
    counts = []
    for block in scored:
        now = OPTIMAL_PATH_TYPE[block[0]["contingency"]]
        before = OPTIMAL_PATH_TYPE[
            "non_protective" if block[0]["contingency"] == "protective" else "protective"
        ]
        n = 0
        for row in block:
            if row["path_type"] == now:
                break
            n += 1 if row["path_type"] == before else 0
        counts.append(n)
    return {"per_block": counts, "mean": float(np.mean(counts)) if counts else None}


def within_window_summary(
    rows: Sequence[Dict[str, Any]],
    window: int = 20,
    exclude_first_reversal: bool = True,
) -> Dict[str, Any]:
    """Area under the adherence curve and asymptotic adherence per block."""
    aligned = align_blocks(rows, window, exclude_first_reversal)
    if not aligned.get("n_blocks"):
        return {}
    auc = aligned["adherence"].mean(axis=1)          # per block
    asymptote = aligned["adherence"][:, -window:].mean(axis=1)
    return {
        "auc_mean": float(auc.mean()),
        "auc_per_block": [float(x) for x in auc],
        "asymptotic_adherence_mean": float(asymptote.mean()),
        "asymptotic_adherence_per_block": [float(x) for x in asymptote],
        "mean_return_per_block": [float(x) for x in aligned["returns"].mean(axis=1)],
    }


def path_type_over_time(
    rows: Sequence[Dict[str, Any]],
    bin_size: int = 50,
) -> Dict[str, List[float]]:
    """Path-type distribution in consecutive bins over the whole run."""
    types = sorted({r["path_type"] for r in rows})
    out: Dict[str, List[float]] = {t: [] for t in types}
    for start in range(0, len(rows), bin_size):
        chunk = rows[start:start + bin_size]
        for t in types:
            out[t].append(sum(r["path_type"] == t for r in chunk) / len(chunk))
    return out


# --------------------------------------------------------------------------
# mood diagnostics
# --------------------------------------------------------------------------

def mood_aligned_to_reversal(
    rows: Sequence[Dict[str, Any]],
    pre: int = 20,
    post: int = 40,
    exclude_first_reversal: bool = True,
) -> Optional[np.ndarray]:
    """Per-episode mean M in a window around each reversal onset.

    Shape (n_reversals, pre + post), column `pre` being the first episode
    under the new contingency.
    """
    blocks = split_blocks(rows)
    if exclude_first_reversal:
        blocks = blocks[1:]
    if not blocks:
        return None

    by_episode = {r["episode"]: r for r in rows}
    windows = []
    for block in blocks:
        onset = block[0]["episode"]
        window = []
        for offset in range(-pre, post):
            row = by_episode.get(onset + offset)
            window.append(row["mean_mood"] if row else np.nan)
        windows.append(window)
    return np.array(windows, dtype=float)


def mood_dip_test(
    rows: Sequence[Dict[str, Any]],
    pre: int = 20,
    post: int = 20,
    exclude_first_reversal: bool = True,
) -> Dict[str, Any]:
    """Does M measurably drop after a reversal?

    **If this fails, the mechanism is not engaging and everything downstream
    is moot** — a mood that does not respond to the contingency change cannot
    be the internal signal that detects it, whatever the adaptation curves do.
    """
    aligned = mood_aligned_to_reversal(rows, pre, post, exclude_first_reversal)
    if aligned is None or aligned.size == 0:
        return {"pass": False, "reason": "no reversal blocks to score"}

    before = np.nanmean(aligned[:, :pre], axis=1)
    after = np.nanmean(aligned[:, pre:], axis=1)
    delta = after - before
    valid = ~np.isnan(delta)
    if not valid.any():
        return {"pass": False, "reason": "no valid mood windows"}

    delta = delta[valid]
    n_dipped = int((delta < 0).sum())
    mean_delta = float(delta.mean())
    # Effect size rather than a p-value: with a handful of reversals per run,
    # significance is not the useful question — magnitude relative to spread is.
    sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    cohens_d = mean_delta / sd if sd > 0 else 0.0

    return {
        "pass": mean_delta < 0 and n_dipped > len(delta) / 2,
        "mean_delta_M": round(mean_delta, 6),
        "mean_M_before": round(float(np.nanmean(before)), 6),
        "mean_M_after": round(float(np.nanmean(after)), 6),
        "n_reversals_scored": int(len(delta)),
        "n_reversals_dipped": n_dipped,
        "cohens_d": round(cohens_d, 4),
        "per_reversal_delta": [round(float(d), 6) for d in delta],
    }


# --------------------------------------------------------------------------
# aggregation across runs
# --------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI over runs (the unit of analysis)."""
    values = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def cohens_d_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Pooled-SD effect size between two groups of run-level values."""
    a = np.asarray([x for x in a if x is not None and not np.isnan(x)])
    b = np.asarray([x for x in b if x is not None and not np.isnan(x)])
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def analyze_study(
    study_dir: str,
    window: int = 20,
    recovery_fraction: float = 0.80,
    exclude_first_reversal: bool = True,
    include_non_acquirers: bool = False,
) -> Dict[str, Any]:
    """Full analysis of a study directory: per-run metrics and group summaries."""
    runs = discover_runs(study_dir)
    per_run: List[Dict[str, Any]] = []

    for run in runs:
        rows = load_schedule(run["schedule_csv"])
        rec = episodes_to_recovery(rows, window, recovery_fraction, exclude_first_reversal)
        entry = {
            **{k: run[k] for k in
               ("agent_type", "seed", "reached_criterion", "acquisition_episodes", "run_dir")},
            "primary": rec[PRIMARY_METRIC],
            "n_censored_blocks": rec["n_censored"],
            "n_scored_blocks": rec["n_scored_blocks"],
            "perseveration": perseveration(rows, exclude_first_reversal)["mean"],
            **within_window_summary(rows, window, exclude_first_reversal),
            "mood_dip": mood_dip_test(rows, window, window, exclude_first_reversal),
            "per_block_recovery": rec["per_block"],
        }
        per_run.append(entry)

    groups: Dict[str, Any] = {}
    for agent_type in sorted({r["agent_type"] for r in per_run}):
        rows_a = [r for r in per_run if r["agent_type"] == agent_type]
        # Runs that never acquired the pre-flip policy have nothing to recover
        # to; they are excluded by default and always counted.
        used = rows_a if include_non_acquirers else [r for r in rows_a if r["reached_criterion"]]
        primaries = [r["primary"] for r in used if r["primary"] is not None]
        mood_pass = [r["mood_dip"].get("pass") for r in used if r["mood_dip"].get("pass") is not None]
        groups[agent_type] = {
            "n_runs": len(rows_a),
            "n_excluded_non_acquirers": len(rows_a) - len(used),
            "n_with_primary": len(primaries),
            "primary_mean": float(np.mean(primaries)) if primaries else None,
            "primary_sd": float(np.std(primaries, ddof=1)) if len(primaries) > 1 else None,
            "primary_ci95": bootstrap_ci(primaries),
            "primary_per_seed": {r["seed"]: r["primary"] for r in used},
            "perseveration_mean": float(np.mean(
                [r["perseveration"] for r in used if r["perseveration"] is not None]
            )) if used else None,
            "auc_mean": float(np.mean(
                [r["auc_mean"] for r in used if r.get("auc_mean") is not None]
            )) if used else None,
            "asymptotic_adherence_mean": float(np.mean(
                [r["asymptotic_adherence_mean"] for r in used
                 if r.get("asymptotic_adherence_mean") is not None]
            )) if used else None,
            "mood_dip_pass_rate": (
                float(np.mean(mood_pass)) if mood_pass else None
            ),
        }

    contrasts = {}
    if "emotional" in groups and "yoked" in groups:
        e = [r["primary"] for r in per_run
             if r["agent_type"] == "emotional" and r["primary"] is not None
             and (include_non_acquirers or r["reached_criterion"])]
        y = [r["primary"] for r in per_run
             if r["agent_type"] == "yoked" and r["primary"] is not None
             and (include_non_acquirers or r["reached_criterion"])]
        contrasts["emotional_vs_yoked"] = {
            "mean_difference": (float(np.mean(e) - np.mean(y)) if e and y else None),
            "cohens_d": cohens_d_between(e, y),
            "note": "negative difference = emotional recovers faster than its yoked control",
        }
    if "yoked" in groups and "baseline" in groups:
        y = [r["primary"] for r in per_run
             if r["agent_type"] == "yoked" and r["primary"] is not None
             and (include_non_acquirers or r["reached_criterion"])]
        b = [r["primary"] for r in per_run
             if r["agent_type"] == "baseline" and r["primary"] is not None
             and (include_non_acquirers or r["reached_criterion"])]
        contrasts["yoked_vs_baseline"] = {
            "mean_difference": (float(np.mean(y) - np.mean(b)) if y and b else None),
            "cohens_d": cohens_d_between(y, b),
            "note": "everything mood-shaped that is NOT the agent's own signal",
        }

    return {
        "study_dir": str(study_dir),
        "primary_metric": PRIMARY_METRIC,
        "window": window,
        "recovery_fraction": recovery_fraction,
        "excluded_first_reversal": exclude_first_reversal,
        "n_runs": len(per_run),
        "groups": groups,
        "contrasts": contrasts,
        "per_run": per_run,
    }


def print_report(analysis: Dict[str, Any]) -> None:
    """Human-readable summary, including the mood pass/fail gate."""
    print("=" * 72)
    print(f"REVERSAL STUDY ANALYSIS: {analysis['study_dir']}")
    print("=" * 72)
    print(f"  Primary metric: {analysis['primary_metric']}")
    print(f"  Recovery window {analysis['window']}, threshold "
          f"{analysis['recovery_fraction']:.0%} of pre-flip level")
    print(f"  Runs analyzed: {analysis['n_runs']}")

    print("\n  " + "-" * 68)
    print(f"  {'agent':<12}{'n':>4}{'excl':>6}{'primary':>10}{'sd':>8}"
          f"{'95% CI':>18}{'persev':>9}")
    print("  " + "-" * 68)
    def _num(value: Optional[float], width: int, digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{'--':>{width}}"
        return f"{value:>{width}.{digits}f}"

    for agent_type, g in analysis["groups"].items():
        ci = g["primary_ci95"]
        ci_s = (f"[{ci[0]:.0f}, {ci[1]:.0f}]"
                if ci and not np.isnan(ci[0]) else "--")
        print(f"  {agent_type:<12}{g['n_runs']:>4}{g['n_excluded_non_acquirers']:>6}"
              f"{_num(g['primary_mean'], 10)}"
              f"{_num(g['primary_sd'], 8)}"
              f"{ci_s:>18}"
              f"{_num(g['perseveration_mean'], 9)}")

    if analysis["contrasts"]:
        print("\n  Contrasts (run-level, effect size with pooled SD):")
        for name, c in analysis["contrasts"].items():
            diff = c["mean_difference"]
            print(f"    {name:<22} diff {diff:+.1f} episodes"
                  f"   d = {c['cohens_d']:+.2f}"
                  if diff is not None else f"    {name}: insufficient data")
            print(f"      {c['note']}")

    print("\n  MOOD DIAGNOSTIC (gate — if mood does not move, nothing downstream means anything)")
    scored_any = False
    for agent_type, g in analysis["groups"].items():
        # Only mood-carrying agents have a mood to test. A baseline "FAIL"
        # would be meaningless — it has no M by construction.
        if agent_type not in MOOD_CARRYING_AGENTS:
            continue
        rate = g["mood_dip_pass_rate"]
        if rate is None:
            continue
        scored_any = True
        verdict = "PASS" if rate >= 0.5 else "FAIL"
        print(f"    {agent_type:<12} M dips after reversal in {rate:.0%} of runs  "
              f"-> {verdict}")
        if agent_type == "emotional" and verdict == "FAIL":
            print("      !! The mood mechanism is not engaging with the contingency "
                  "change. Adaptation differences cannot be attributed to mood.")
    if not scored_any:
        print("    no mood-carrying runs scored yet — gate NOT evaluated "
              "(this is not a pass)")
    print("=" * 72)
