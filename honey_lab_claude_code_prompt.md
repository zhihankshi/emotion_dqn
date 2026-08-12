# Honey Lab — Implementation Brief: Mood Mechanism Audit, Controls, and Reversal Experiment

## Context

This repo tests whether a computational mood signal (Emanuel & Eldar 2023, "Emotions as Computations")
improves DQN learning in visual mazes versus a matched baseline DQN.

Current formulation:
- Baseline target: `Q_target = Q + η·δ`
- Emotional target: `Q_target = Q + η·δ + (1-η)·M`
- Mood update: `M(t+1) = M(t) + (1-λ)·(η·δ(t) - M(t))`, defaults η=0.9, λ=0.8, M clipped to [-1,1], persists across episodes.

**We are reframing the project.** We are no longer asking "does mood-DQN beat DQN." We are asking:
*when Emanuel & Eldar's mood mechanism is instantiated in a real learning agent, does it produce the
adaptive benefit the theory claims, and under what environmental conditions?* The theory predicts mood
helps via generalization in **stochastic or non-stationary** environments. Our current mazes are small,
deterministic, and stationary — close to the worst case for the mechanism. A null result there may be a
finding, not a bug. The decisive experiment is repeated contingency reversal.

All work below must be **config-gated with defaults that preserve current behavior**, so existing results
remain reproducible.

---

## Task 0 — Audit first, implement nothing yet

Read the code and report back before making changes. Do not modify anything in this step.

1. **What δ does the mood tracker actually consume?** Trace it precisely. Specifically determine whether
   `MoodTracker` inside `EmotionalDQNAgent` is fed:
   - (a) the mean/sum of TD errors over a sampled replay batch, or
   - (b) the online TD error of the transition the agent just experienced, in time order.

   This matters enormously. Emanuel & Eldar's mood integrates the agent's *experienced* value updates in
   temporal order. Option (a) integrates off-policy, temporally scrambled transitions and is **not** the
   quantity the theory defines. Report which one it is, with file and line references.

2. **Is δ signed or absolute?** Signed δ is valence-like (the theory's intent). Absolute δ is
   surprise/arousal-like. Report which.

3. **Report the empirical scale of things.** Instrument a short run and report: reward magnitudes per maze,
   the observed distribution of δ, the observed range and standard deviation of M, and how often M hits the
   [-1,1] clip. If `(1-η)·M` is two orders of magnitude smaller than typical `η·δ`, say so explicitly —
   that quantifies why we see no behavioral difference.

4. **Confirm the baseline and emotional agents are truly matched** on network init, seed handling, optimizer,
   buffer, target-net sync, and ε schedule. List any asymmetry you find.

5. Report how the replay buffer, ε schedule, and target network currently behave across a phase boundary
   in `scripts/train_transfer.py`.

**Stop after the audit and report. Wait for confirmation before proceeding.**

---

## Task 1 — Make the mood signal source explicit and configurable

Add a config option `mood.delta_source` with values:
- `replay_batch` — current behavior (whatever the audit found), preserved as default if that's what exists
- `online` — mood updated from the agent's experienced transition, in time order, independent of replay sampling

Implement `online` if it does not already exist. Also add `mood.delta_signed: true|false`.

Both variants must be runnable so the δ source becomes an explicit experimental variable rather than an
unexamined implementation choice. Document the difference in a docstring citing the theoretical motivation.

---

## Task 2 — Yoked mood control agent (highest priority)

This is the control that makes any result interpretable. Without it, a positive result cannot be
distinguished from "we injected temporally correlated noise into the Q-targets" or "we changed the
effective step size."

Implement `YokedDQNAgent`: identical to `EmotionalDQNAgent` in every respect, except that the mood value
used in the target is **decoupled from its own TD errors**.

Two yoking modes, both configurable:
- `yoked.mode: replay_trace` — load a recorded M(t) trace from a completed `EmotionalDQNAgent` run
  (different seed) and consume it step by step. Exactly matches magnitude and autocorrelation.
- `yoked.mode: ou_process` — generate an Ornstein–Uhlenbeck process with mean, variance, and
  autocorrelation fitted to the recorded mood traces.

For `replay_trace`, add mood-trace recording to `EmotionalDQNAgent` (per-step M, written to disk alongside
episode logs) and a loader that handles length mismatch by holding the final value.

Every comparison from here forward is three-way: **baseline vs emotional vs yoked**.

---

## Task 3 — Repeated contingency reversal environment

Currently `scripts/train_transfer.py` does a single source→target transfer. A single flip gives one
adaptation event per run and very low statistical power. Replace with repeated reversals to get
repeated measures within each run.

Add `scripts/train_reversal.py`:

**Phase A — Acquisition.** Train on `shield_trap` (or `shield_trap_easy`, configurable) until a competence
criterion is met: ≥80% `shield_route` path type over a 50-episode rolling window. Cap at 1000 episodes.
Log whether each run reached criterion; runs that never do are flagged, not silently included.

**Phase B — Reversal block.** Alternate the contingency every `K` episodes for `R` reversals
(target R = 8–10). Reversal = swap between `shield_trap` (shield reduces trap cost) and
`shield_avoidance` (shield does not). **Geometry, sprites, and start/goal positions must be identical
across the two configs** — only the shield's effect on trap cost changes. Verify this and report any
visual difference, since a visible cue would let the CNN detect the switch directly and confound everything.

**Critical constraints — these are the experiment, not implementation details:**

- **Do not reset or bump ε at reversals.** Use a constant ε floor (default 0.05, configurable) throughout
  Phase B. Bumping ε signals the change externally and destroys the hypothesis, since the claim under test
  is that mood is the *internal* signal that detects the change.
- **Do not flush or clear the replay buffer at reversals.** Same information leak.
- **Shrink the replay buffer** so it turns over within roughly one reversal period (default 12,000,
  configurable). At 50k the buffer spans the whole run and both agents are dominated by stale transitions.
  Apply identically to all three agent types.
- **Do not reset the target network at reversals.**
- Mood must persist across reversals — no resetting M.

---

## Task 4 — Pilot to calibrate the reversal period

Before running the full study, add `scripts/pilot_recovery.py`:

Run Phase A to criterion, execute a **single** flip, and measure **episodes-to-recovery** — episodes until
the rolling success/path-type metric returns to 80% of its pre-flip level. Run with 5 seeds for baseline
and emotional.

Report median and IQR of recovery time. We will set `K ≈ 1.5–2× median recovery`. If median recovery on
`shield_trap` exceeds ~300 episodes, recommend switching the reversal study to `shield_trap_easy` — the
reversal experiment tests adaptation, not maze difficulty, so using the easier maze is legitimate and
should be stated as such.

Report the estimated wall-clock and total-episode budget for the full study at the calibrated K:
3 agent types × N seeds × (Phase A + R×K) episodes.

---

## Task 5 — Metrics and logging

Add to `utils/`:

1. **Reversal-aligned analysis.** Slice episode logs into windows aligned to reversal onset, then average
   across reversals within a run and across runs. Produce recovery curves with confidence bands, three
   agent types overlaid. This is the primary figure.
2. **Primary pre-registered metric:** mean episodes-to-recovery, averaged across reversals 2–R
   (excluding the first, which is confounded with end-of-acquisition). Fix this before looking at results.
3. **Secondary metrics:** perseveration (episodes still taking the old-optimal route post-flip), area under
   the learning curve within each reversal window, asymptotic performance within window, path-type
   distribution over time.
4. **Mood diagnostics.** Plot M(t) aligned to reversal onset. **If M does not measurably dip after a
   reversal, the mechanism is not engaging and everything downstream is moot** — surface this as an
   explicit pass/fail printout, not just a plot.
5. Log per-step M, per-episode mean/min/max M, and clip-saturation frequency for all mood-carrying agents.

---

## Task 6 — Hyperparameter hygiene

Add sweep support to `compare_agents.py`:

- Sweep η ∈ {0.5, 0.7, 0.9} and λ ∈ {0.5, 0.8, 0.95} for the emotional agent.
- Sweep the **baseline's** learning rate over a comparable range. Without this, any emotional win may just
  be an effective-step-size change wearing a costume.
- Support ≥20 seeds per condition and report per-seed variance, not just means. Deep RL seed variance is
  large enough that 3–5 seeds cannot detect a small effect.
- Report effect sizes with confidence intervals, not just significance.

---

## Task 7 — Documentation

Write `EXPERIMENT_DESIGN.md` covering: the reframed research question, the reversal protocol, the role of
the yoked control, the pre-registered primary metric, and the constraints in Task 3 with their rationale
(so no one "helpfully" adds an ε reset later).

Include a **falsification section**: if the emotional agent does not beat the yoked control on the primary
metric across the reversal study with ≥20 seeds, the conclusion is that the mood mechanism as formulated
does not confer the claimed adaptive benefit in this setting. That is a reportable result under the
reframed question, and we write it up as such rather than continuing to search for a maze where it works.

---

## Working order

Task 0 (audit, report, stop) → 1 → 2 → 3 → 4 (pilot, report, stop) → 5 → 6 → 7.

Stop and report after Task 0 and after Task 4. Keep all changes config-gated with behavior-preserving
defaults. Do not modify baseline agent behavior at any point.
