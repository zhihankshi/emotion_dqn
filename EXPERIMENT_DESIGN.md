# Experiment Design — Mood as a Computational Signal in Reversal Learning

This document is the reference for what the reversal study tests, how it is run, and
what would count as a negative result. Several design constraints look like bugs to a
reader trying to be helpful. They are not. Sections marked **Do not "fix" this** explain
why.

---

## 1. The research question

The project no longer asks *"does mood-DQN beat DQN?"* That framing invites tuning
until the favoured agent wins.

The question is:

> When Emanuel & Eldar's mood mechanism is instantiated in a real learning agent, does
> it produce the adaptive benefit the theory claims, and under what environmental
> conditions?

The theory predicts mood helps via generalization in **stochastic or non-stationary**
environments. Small, deterministic, stationary mazes are close to the worst case for
the mechanism, so a null result there is a finding about scope, not a bug. The decisive
test is therefore **repeated contingency reversal**, which manufactures
non-stationarity while holding everything else fixed.

### The learning rules

| agent | Q-target | where M comes from |
|---|---|---|
| baseline | `Q + η·δ` | — |
| emotional | `Q + η·δ + (1−η)·M` | its own TD errors |
| yoked | `Q + η·δ + (1−η)·M` | **another agent's** recorded M, or an OU surrogate |

Mood update: `M ← M + (1−λ)·(η·δ − M)` = `λ·M + (1−λ)·η·δ`, clipped to `mood_bounds`,
persisting across episodes. **λ is the retention per mood update** — the half-life of M
is measured in mood updates, not episodes or gradient steps.

---

## 2. Why the yoked control is the centre of the design

The emotional agent adds a term to every Q-target. If it beats the baseline, at least
three explanations survive:

1. mood carries useful information about the agent's own learning progress — **the
   hypothesis**;
2. adding a temporally correlated perturbation to Q-targets helps regardless of origin
   — noise injection or annealing;
3. adding a nonzero term changed the effective step size.

`YokedDQNAgent` (`agents/yoked_dqn.py`) is identical to the emotional agent in every
respect — same network, optimizer, buffer, ε schedule, seeding, target-net sync, same
learning rule — except that M is supplied from outside and never touches its own δ. It
reproduces (2) and (3) while destroying (1).

    emotional − yoked  =  the effect of mood being the agent's OWN signal
    yoked − baseline   =  everything else that is mood-shaped

Two yoking modes: `replay_trace` (consume a per-step M(t) trace recorded by an
emotional run with a **different seed**, holding the final value if the donor is
shorter) and `ou_process` (AR(1) with μ, σ, φ fitted to recorded traces).

**Every comparison from here forward is three-way.** A baseline-vs-emotional result
alone is not interpretable and should not be reported as one.

> **Verification that the match is real:** a yoked agent given a constant M = 0 produces
> trajectories bit-identical to the baseline. The OU process draws from its own RNG
> stream, never the shared one, precisely so the agents' ε-greedy and replay sampling
> sequences stay aligned.

---

## 3. The reversal protocol

Driver: `scripts/train_reversal.py` (single run), `scripts/run_reversal_study.py`
(full study).

### Phase A — acquisition

Train under the *protective* contingency until competence: ≥ `criterion_rate` (0.80)
`shield_route` episodes over a rolling `criterion_window` (50), capped at
`max_acquisition_episodes` (1000). The criterion cannot be met before ε reaches its
floor (`min_acquisition_episodes` defaults to the ε decay horizon) — competence measured
under heavy exploration is not competence.

Runs that never reach criterion are **flagged in the manifest and excluded from the
primary metric**, never silently pooled.

### Phase B — reversal block

Flip the contingency every `K` episodes, `R` times (default R = 8, K = 400 — see §6).

Both contingencies are built from the **same maze file** with one reward key changed:

| contingency | `trap_with_shield` |
|---|---|
| `protective` | as authored (shield reduces trap cost) |
| `non_protective` | `--non_protective_trap` (shield does not protect, or actively hurts) |

Nothing else differs — geometry, sprites, start/goal, step cost, pickup value, timeout,
`max_steps` are literally the same file.

### Two checks that run before training, and abort it

1. **Visual identity.** Every reachable state is rendered under both contingencies and
   compared pixel-by-pixel. A visible cue would let the CNN detect the switch directly
   and confound everything. Verified: 44 states on `shield_trap`, 24 on
   `shield_trap_easy`, zero pixel differences.
2. **The reversal actually reverses.** Canonical routes are evaluated under both
   contingencies and the run aborts unless the *optimal route flips*.

The second check exists because the obvious configuration silently fails it. Setting
`trap_with_shield = trap_no_shield` makes the shield useless but leaves the shield
*detour* weakly optimal, because the pickup bonus exceeds the detour's step cost:

    shield_trap_easy  non_protective: shield_route −23  vs  direct −27
    shield_trap       non_protective: shield_route −47  vs  direct −48

The optimal path type never changes, so there is no behavioural adaptation and
episodes-to-recovery is undefined. `--non_protective_trap` fixes this by making the
shield actively harmful. At −60 on `shield_trap_easy` (with `trap_no_shield=-20`) the
margins are near-symmetric.

> `shield_trap` vs `shield_avoidance` is **not** a valid reversal pair. Those mazes also
> differ in `trap_no_shield` (−50 vs −5) and `shield_pickup` (+5 vs 0), confounding the
> contingency flip with trap severity and pickup value.

---

## 4. Constraints — **Do not "fix" these**

These are the experiment. Each one, if "helpfully" relaxed, destroys the hypothesis
under test: that mood is the *internal* signal by which the agent detects the change.

| constraint | why |
|---|---|
| **ε is never reset or bumped at a reversal.** Constant floor (0.05) through all of Phase B. | Bumping ε is an *external* announcement that the world changed. The agent must detect it internally or not at all. ε is computed from the global episode index so there is no per-phase schedule to reset. |
| **The replay buffer is never flushed.** | Same information leak: flushing tells the agent its past is invalid. |
| **The buffer is small (12,000) so it turns over within ~one reversal period.** Applied identically to all three agent types. | At 50k the buffer spans the whole run and every agent trains mostly on stale, wrong-contingency data, masking the adaptation being measured. The run prints measured steps/episode against K and warns if the buffer spans > 1.5 periods. |
| **The target network is never reset.** | Same leak, and it would discard the value estimates whose revision *is* the adaptation. |
| **Mood persists across reversals.** M is never reset. | A mood reset would substitute an external cue for the internal signal. |
| **One agent object for the entire run.** | Rebuilding the agent silently resets optimizer state, buffer, and target net at once — this is how `train_transfer.py` leaked on all four channels. |

Verified by instrumenting a live run: exactly one ε value across Phase B,
`ReplayBuffer.clear()` called 0 times, `MoodTracker.reset()` called 0 times. The
manifest records all four as explicit `False` flags so a run can be audited without
rereading the code.

---

## 5. Metrics

Analysis: `utils/reversal_analysis.py`, figures: `visualize_reversal.py`.

Everything is aligned to **reversal onset**, averaged within a run across reversals,
then across runs. The unit of analysis is the **run**, so a run with more reversals
cannot outvote one with fewer.

### Pre-registered primary metric

> **Mean episodes-to-recovery, averaged across reversals 2..R.**

Fixed before looking at results. Reversal 1 is excluded because it is confounded with
the end of acquisition — the agent has just left a criterion-terminated phase, its
buffer is full of acquisition data, and ε has only just settled.

Recovery is defined on **adherence to the currently-optimal route**, not raw return: the
two contingencies have very different optimal returns, so "80% of the pre-flip return"
would compare incomparable quantities. A block that never recovers is **right-censored
at block length and counted**, never dropped — dropping it would bias the mean toward
whichever agent recovers at all.

### Secondary metrics

- **Perseveration** — episodes still taking the old-optimal route after the flip.
- **Area under the adherence curve** within each reversal window.
- **Asymptotic adherence** within window.
- **Path-type distribution** over time.

### Reporting

Effect sizes (Cohen's d) with bootstrap confidence intervals over runs. Per-seed values
are always written out. No significance stars: with this many seeds and this much
variance, magnitude relative to spread is the useful question.

### The mood gate — check this first

`mood_dip_test` asks whether M measurably drops after a reversal, and prints an explicit
**PASS/FAIL**.

> **If M does not dip after a reversal, the mechanism is not engaging and everything
> downstream is moot.** A mood that does not respond to the contingency change cannot be
> the internal signal that detects it, whatever the recovery curves show. Read this
> before reading the primary metric.

Per-step M is logged to `mood_trace.csv`; per-episode mean/min/max M and clip saturation
go to the episode CSV. Clip saturation matters: raw maze rewards span roughly ±55
against a default ±1 mood clip, so M can spend its time pinned at the bounds. Use
`--reward_scale` or `--mood_clip_range` to bring them into the same range.

---

## 6. Calibration and known parameter values

From `scripts/pilot_recovery.py` on `shield_trap_easy` (2 seeds × baseline/emotional,
600-episode post-flip cap, `bootstrap_on_truncation` on):

- Median episodes-to-recovery **192** (without the truncation fix: 253)
- → **K = 400** (≈2× median), R = 8
- Median acquisition 120–150 episodes
- ~0.4 s/episode; 3 agent types × 20 seeds ≈ 200k episodes ≈ 24 h serial

Pilot caps must exceed expected recovery or every run censors — the first three pilot
configurations censored purely because a 150–200 episode cap was shorter than recovery.

### δ source is an experimental variable, not an implementation detail

`mood.delta_source` selects what feeds M:

- `online` — one update per env step from the experienced transition, in time order.
  **Theory-faithful**: Emanuel & Eldar's mood integrates the agent's own value updates
  in temporal order.
- `batch_mean` — one update per gradient step from the mean batch δ.
- `batch_sequential` — **default, historical**: one update per batch element, i.e. 32
  per gradient step. Effective retention per gradient step is λ^32 ≈ 8e−4 at λ = 0.8, so
  M is essentially rebuilt from the current batch each step and carries almost no
  history. Approximating an *intended* retention of 0.8 under this mode needs
  λ ≈ 0.993.

Measured on `shield_trap_easy`: M autocorrelation at lag 1 env step is **0.21** under
`batch_sequential` versus **0.73** under `online`. Under the default, M is close to
white noise; runs using it are not testing a slow-moving mood.

### Time-limit bootstrapping

`bootstrap_on_truncation` (default off) separates the two meanings of "done": a timeout
ends the episode but is not a terminal state, so zeroing the bootstrap there hides the
cost of continuing and makes stalling look cheap (Pardo et al., *Time Limits in
Reinforcement Learning*). Enable it for the reversal study; it applies identically to
all agent types, so the matched comparison is unaffected.

---

## 7. Falsification

**If the emotional agent does not beat the yoked control on the primary metric across
the reversal study with ≥ 20 seeds, the conclusion is that the mood mechanism as
formulated does not confer the claimed adaptive benefit in this setting.**

That is a reportable result under the reframed question, and it is written up as such.
It is not a cue to search for a maze, reward scale, or hyperparameter setting where the
effect appears.

Specifically, the following are **not** licensed responses to a null result:

- adding an ε bump, buffer flush, or target reset at reversals to "help adaptation";
- switching the primary metric after seeing the data, or dropping reversal 1's exclusion;
- excluding censored blocks, or non-acquiring runs beyond the pre-specified rule;
- reporting baseline-vs-emotional without the yoked arm;
- tuning η or λ post hoc and reporting the winning cell as the result.

The sweep in §5 exists to characterize the mechanism's parameter dependence **as a
whole**, reported as a grid with per-seed spread — not to select a favourable cell.

A null result would still leave two publishable claims: a quantitative account of why
the mechanism cannot matter at the default settings (the mood term is 17–24× smaller
than the TD term and near-white at lag 1), and a reusable protocol with a control that
makes any future positive result interpretable.
