"""
Yoked mood control agent.

This is the control that makes any emotional-agent result interpretable.
``EmotionalDQNAgent`` adds ``(1 - η)·M`` to every Q-target, where M is built
from the agent's own TD errors. A win over the baseline therefore has at least
three candidate explanations:

1. mood carries useful information about the agent's own learning progress
   (the hypothesis),
2. adding a temporally correlated perturbation to Q-targets helps regardless
   of where it came from (noise injection / annealing),
3. adding a nonzero term to the target changed the effective step size.

``YokedDQNAgent`` is identical to ``EmotionalDQNAgent`` in every respect —
same network, optimizer, buffer, ε schedule, seeding, target-net sync, and the
same ``Q_target = Q + η·δ + (1 - η)·M`` rule — except that M is **supplied from
outside** and never touches its own δ. It therefore reproduces (2) and (3)
while destroying (1). Emotional minus yoked is the effect attributable to mood
being *the agent's own signal*; yoked minus baseline is everything else.

Two yoking modes:

``replay_trace``
    Consume a per-step M(t) trace recorded by a completed ``EmotionalDQNAgent``
    run (a different seed). Matches magnitude, distribution, and
    autocorrelation exactly, because it *is* a real mood trace — just one
    belonging to another agent's learning history.

``ou_process``
    An Ornstein–Uhlenbeck / AR(1) process with mean, variance, and
    autocorrelation fitted to recorded mood traces. Matches those three moments
    but carries no learning history at all. Useful when no length-matched trace
    is available, and as a check that ``replay_trace`` results are not driven
    by some finer structure of the donor trace.

RNG note: the OU process draws from its **own** ``np.random.Generator``, never
from the global ``np.random`` stream the agents share for ε-greedy and replay
sampling. Without this, the yoked agent would consume RNG the other two do not
and its action/sampling sequence would silently diverge — breaking the match it
exists to provide.
"""
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .emotional_dqn import EmotionalDQNAgent

YOKED_MODES = ("replay_trace", "ou_process")

# What a replayed trace does once the recipient outlives it.
EXHAUSTION_MODES = ("reflect", "hold")


def load_mood_trace(path: str) -> np.ndarray:
    """Load a per-step mood trace written by ``EmotionalDQNAgent.save_mood_trace``.

    Accepts either the CSV itself or a run directory containing
    ``mood_trace.csv``.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "mood_trace.csv"
    if not p.exists():
        raise FileNotFoundError(f"No mood trace at {p}")

    values: List[float] = []
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            values.append(float(row["mood"]))
    if not values:
        raise ValueError(f"Mood trace {p} is empty")
    return np.asarray(values, dtype=np.float64)


def load_mood_traces(paths: Sequence[str]) -> List[np.ndarray]:
    return [load_mood_trace(p) for p in paths]


class TraceMoodSource:
    """Replays a recorded M(t) trace one value per environment step.

    Donor and recipient lengths never match: acquisition is criterion-
    terminated, so a yoked run routinely outlives the emotional run it is
    yoked to. ``exhaustion`` decides what happens past the end of the trace.

    ``reflect`` (default)
        Mirror the trace at its endpoint and walk back: M(T+k) = M(T-k). The
        join is continuous, and the padding keeps the donor's marginal
        distribution and local autocorrelation, so the control stays a mood
        *shaped* signal for the whole run.

    ``hold`` (legacy)
        Pin M at the donor's final value. This is what the original
        implementation did, and it silently degrades the control into a
        constant offset — worse, into a constant at whatever extreme the donor
        happened to end on, which shows up as a spike in clip saturation. Kept
        only so earlier studies remain reproducible.

    Wrapping to the start is deliberately not offered: it injects a jump
    discontinuity at every wrap.
    """

    mode = "replay_trace"

    def __init__(
        self,
        trace: np.ndarray,
        source_path: Optional[str] = None,
        exhaustion: str = "reflect",
    ):
        if len(trace) == 0:
            raise ValueError("Mood trace is empty")
        if exhaustion not in EXHAUSTION_MODES:
            raise ValueError(
                f"exhaustion must be one of {EXHAUSTION_MODES}, got {exhaustion!r}"
            )
        self.trace = np.asarray(trace, dtype=np.float64)
        self.source_path = source_path
        self.exhaustion = exhaustion
        self.index = 0
        self.n_held = 0  # steps served from padding rather than the trace itself

    def __len__(self) -> int:
        return len(self.trace)

    def _reflected(self, index: int) -> float:
        """Triangle-wave index into the trace: forward, back, forward, ..."""
        n = len(self.trace)
        if n == 1:
            return float(self.trace[0])
        period = 2 * n - 2
        pos = index % period
        return float(self.trace[pos if pos < n else period - pos])

    def next(self) -> float:
        if self.index < len(self.trace):
            value = float(self.trace[self.index])
        else:
            self.n_held += 1
            value = (float(self.trace[-1]) if self.exhaustion == "hold"
                     else self._reflected(self.index))
        self.index += 1
        return value

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.source_path,
            "length": len(self.trace),
            "consumed": self.index,
            "exhaustion": self.exhaustion,
            "held_final": self.n_held,          # steps served from padding
            "frac_padded": (round(self.n_held / self.index, 4)
                            if self.index else 0.0),
            "trace_mean": float(self.trace.mean()),
            "trace_std": float(self.trace.std()),
        }


class OUMoodSource:
    """Discrete Ornstein–Uhlenbeck (AR(1)) mood surrogate.

        M_{t+1} = μ + φ·(M_t - μ) + ε_t,   ε_t ~ N(0, σ_ε²)

    with σ_ε = σ·sqrt(1 - φ²) so the stationary standard deviation is σ. φ is
    the lag-1 autocorrelation; the continuous-time OU relaxation rate is
    θ = -ln(φ) per step.
    """

    mode = "ou_process"

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.3,
        phi: float = 0.9,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None,
        fitted_from: Optional[Sequence[str]] = None,
    ):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.phi = float(np.clip(phi, -0.999999, 0.999999))
        self.bounds = bounds
        self.fitted_from = list(fitted_from) if fitted_from else None
        # Private stream: must not perturb the shared np.random sequence.
        self.rng = np.random.default_rng(seed)
        self.value = self.mu
        self.n_drawn = 0

    @classmethod
    def fit(
        cls,
        traces: Sequence[np.ndarray],
        bounds: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None,
        fitted_from: Optional[Sequence[str]] = None,
    ) -> "OUMoodSource":
        """Fit μ, σ and lag-1 autocorrelation φ to recorded mood traces.

        φ is pooled as the ratio of summed lag-1 autocovariance to summed
        variance across traces, so long traces dominate proportionally and a
        short trace cannot swing the estimate.
        """
        traces = [np.asarray(t, dtype=np.float64) for t in traces if len(t) > 1]
        if not traces:
            raise ValueError("Need at least one trace of length > 1 to fit")

        allv = np.concatenate(traces)
        mu = float(allv.mean())
        sigma = float(allv.std())

        num = den = 0.0
        for t in traces:
            c = t - mu
            num += float(np.dot(c[:-1], c[1:]))
            den += float(np.dot(c[:-1], c[:-1]))
        phi = num / den if den > 0 else 0.0

        return cls(mu=mu, sigma=sigma, phi=phi, bounds=bounds, seed=seed,
                   fitted_from=fitted_from)

    def next(self) -> float:
        eps_sd = self.sigma * np.sqrt(max(0.0, 1.0 - self.phi ** 2))
        raw = self.mu + self.phi * (self.value - self.mu) + self.rng.normal(0.0, eps_sd)
        # Clipped like a real mood, so the surrogate saturates the same way.
        self.value = float(np.clip(raw, self.bounds[0], self.bounds[1]))
        self.n_drawn += 1
        return self.value

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "mu": self.mu,
            "sigma": self.sigma,
            "phi": self.phi,
            "theta_per_step": float(-np.log(abs(self.phi))) if self.phi else float("inf"),
            "bounds": list(self.bounds),
            "drawn": self.n_drawn,
            "fitted_from": self.fitted_from,
        }


class YokedDQNAgent(EmotionalDQNAgent):
    """Emotional DQN whose mood is decoupled from its own TD errors.

    Everything except the origin of M is inherited unchanged, so the three-way
    baseline / emotional / yoked comparison stays matched by construction.

    Args:
        mood_source: a :class:`TraceMoodSource` or :class:`OUMoodSource`.
            Advanced once per environment step, immediately after the gradient
            update — the same point in the loop where the emotional agent
            integrates its own δ — so M enters the target with identical lag.
        Remaining args are those of :class:`EmotionalDQNAgent`. ``lambda_mood``
        and ``delta_source`` are accepted and recorded but have no effect: no
        δ is ever integrated.
    """

    def __init__(self, *args, mood_source=None, **kwargs):
        super().__init__(*args, **kwargs)
        if mood_source is None:
            raise ValueError(
                "YokedDQNAgent requires a mood_source (TraceMoodSource or "
                "OUMoodSource) — that is the whole point of the control"
            )
        self.mood_source = mood_source
        self.yoked_mode = mood_source.mode
        # Start from the source's first value rather than 0, so the very first
        # targets already carry the imposed mood.
        self.mood_tracker.mood = mood_source.next()

    def _accumulate_mood(self, td_error_batch, td_error: float) -> float:
        """No-op: the yoked agent's mood ignores its own δ entirely."""
        return self.mood_tracker.get_mood()

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: Optional[Sequence[int]] = None,
    ) -> Optional[Dict[str, float]]:
        metrics = super().step(
            state, action, reward, next_state, done, next_valid_actions
        )
        # Advance after the gradient update, matching the emotional agent's
        # "use M_t for the target, accumulate afterwards" ordering.
        self.mood_tracker.mood = self.mood_source.next()
        if self.record_mood_trace and self.mood_trace:
            self.mood_trace[-1] = self.mood_tracker.get_mood()
        return metrics

    def get_metrics(self) -> Dict[str, Any]:
        metrics = super().get_metrics()
        metrics["yoked_mode"] = self.yoked_mode
        return metrics

    def describe_mood_source(self) -> Dict[str, Any]:
        return self.mood_source.describe()

    def save_checkpoint(self, path: str, episode: int) -> None:
        super().save_checkpoint(path, episode)
        # Tag the agent type honestly so downstream analysis cannot mistake a
        # yoked checkpoint for an emotional one.
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt["agent_type"] = "yoked"
        ckpt["yoked_mode"] = self.yoked_mode
        torch.save(ckpt, path)


def build_mood_source(
    mode: str,
    trace_paths: Optional[Sequence[str]] = None,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = None,
    ou_params: Optional[Dict[str, float]] = None,
    exhaustion: str = "reflect",
) -> Any:
    """Construct the mood source for a yoked run.

    Args:
        mode: 'replay_trace' or 'ou_process'.
        trace_paths: donor mood traces (CSV files or run dirs). Required for
            replay_trace; for ou_process, used to fit μ/σ/φ unless explicit
            ou_params are given.
        bounds: mood clip range, matched to the emotional agent's.
        exhaustion: 'reflect' | 'hold' — what a replayed trace does once the
            recipient outlives it. See :class:`TraceMoodSource`.
        seed: seed for the OU process's private RNG.
        ou_params: explicit {'mu', 'sigma', 'phi'}, skipping the fit.
    """
    if mode not in YOKED_MODES:
        raise ValueError(f"yoked mode must be one of {YOKED_MODES}, got {mode!r}")

    if mode == "replay_trace":
        if not trace_paths:
            raise ValueError(
                "yoked mode 'replay_trace' needs at least one donor mood trace "
                "(--yoked_trace); run the emotional agent first"
            )
        if len(trace_paths) > 1:
            traces = load_mood_traces(trace_paths)
            longest = int(np.argmax([len(t) for t in traces]))
            return TraceMoodSource(traces[longest],
                                   source_path=str(trace_paths[longest]),
                                   exhaustion=exhaustion)
        return TraceMoodSource(load_mood_trace(trace_paths[0]),
                               source_path=str(trace_paths[0]),
                               exhaustion=exhaustion)

    if ou_params:
        return OUMoodSource(bounds=bounds, seed=seed, **ou_params)
    if not trace_paths:
        raise ValueError(
            "yoked mode 'ou_process' needs either --yoked_trace (to fit "
            "mu/sigma/phi) or explicit ou_params"
        )
    return OUMoodSource.fit(
        load_mood_traces(trace_paths), bounds=bounds, seed=seed,
        fitted_from=[str(p) for p in trace_paths],
    )
