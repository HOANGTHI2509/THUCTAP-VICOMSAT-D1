# OperationalGuard progress checkpoint

Checkpoint scope: causal Smooth-Tracking operational layer only. Random Forest and Ground Truth were not modified or retrained.

## Implemented

- Separate `OperationalGuard` between classifier evidence and Kalman. RF is advisory and never selects Q/R directly.
- Per-vehicle state, SegmentID reset, and reset for TimeGap greater than 30 minutes.
- Long-lived causal excursions with published CleanFuel soft/horizontal hold and independent median-3 + EWMA ShadowFuel.
- States: `STABLE`, `DOWN_EXCURSION`, `PENDING_DOWNWARD`, `PENDING_UPWARD`, `LOW_PLATEAU_UNCERTAIN`, `GRADUAL_TRACKING`, `PERSISTENT_TREND_ESCAPE`, `REBOUND_RECOVERY`, `U_SHAPE_CONFIRMED`, `OSCILLATION`, `DOWNWARD_CONFIRMED`, `UPWARD_CONFIRMED`, `BASELINE_REACQUISITION`.
- Baseline-deviation thresholds use `max(percent * known capacity, k * causal MAD noise)`; unknown capacity uses absolute safety floor plus noise. Invalid declared capacity downgrades to `UNKNOWN_CAPACITY_MODE` with a warning.
- Speed-conditioned online expected-rate profiles: vehicle+bin, vehicle, fleet+bin, fleet, safe prior. Bins are STOPPED/LOW/MID/HIGH using median of the last three speeds.
- Rebound/pullback cancel, elapsed-time recovery lock, strong persistent opposite-event escape, single-sample innovation gating, persistent-trend escape, strong-UP fast path, time-scaled confirmed Q/R ramp, and robust baseline reacquisition.
- Required debug columns are propagated through DataFrame/dashboard; Data Inspector shows the operational diagnostics.
- State persistence includes excursion, recovery, trend, transition, innovation and capacity memory.

## Causal leakage found and fixed

`run_topic1_filter` previously used one engine for the full-segment classifier prepass and published filtering. Although classifier feature extraction was sequential, that prepass could populate OperationalGuard/rate-profile memory using later rows before the first published CleanFuel row was processed.

It now creates two independent objects:

- `classifier_engine`: RF feature/state calculation only.
- `filter_engine`: published OperationalGuard, rate profiles and Kalman state only.

No context, rate profile, operational memory, or Kalman state is shared. Two critical tests prove prefix invariance and dashboard causal-batch equivalence with direct sample streaming.

## Files changed/created

- `src/core/filters/smooth_tracking/operational_guard.py`
- `src/core/filters/smooth_tracking/config.py`
- `src/core/filters/smooth_tracking/state.py`
- `src/core/filters/smooth_tracking/engine.py`
- `src/core/filters/smooth_tracking/dataframe.py`
- `src/dashboard/dashboard_data.py`
- `src/dashboard/app_dashboard_tienxuly.py`
- `src/service/state_store.py`
- `tests/test_operational_guard_golden.py`
- `tests/test_dashboard_topic1.py`
- `scripts/replay_operational_guard_fleet.py` (provisional replay utility)
- `docs/OPERATIONAL_GUARD_PROGRESS.md`

## Test status

- Critical causal tests: **2 passed**.
  - Prefix invariance.
  - Streaming equals dashboard causal batch.
- Compile check for OperationalGuard, engine and dashboard orchestration: **passed**.
- Operational Golden tests before the final isolation-only patch: **24 passed**.
- Dashboard/API/state/concurrency critical group before the final isolation-only patch: **41 passed**.
- Last full regression before final causal isolation: **121 passed, 14 failed**. The 14 failures are unchanged exact legacy curve/snapshot expectations (12 real-segment clean curves plus 2 restored-filter exact curves). Behavioral contracts passed. Snapshots were deliberately not updated pending review.
- Full regression has **not** been rerun after the final causal-isolation patch.
- The 9-vehicle replay generated earlier is provisional; it has **not been rerun and reviewed after final causal isolation**, so fleet and per-capacity acceptance remain incomplete.

## Current defaults (key operational values)

- Reset gap: 30 min; robust noise: `max(0.10, 1.4826 * MAD(delta_raw))`.
- Deviation: LOW_MOTION 1.2%, MOVING 2.0%; candidate 1.8%; confirm 3.5%; strong shift 4.5%; rebound cancel 0.60.
- Unknown absolute safety floor: 0.5 L; no trusted default 200 L.
- New-baseline elapsed acceptance: 30 min; recovery: minimum 2 samples and 2 elapsed minutes, with no fixed-beat expiry.
- Q/R: stable 0.05/28; pending 0.015/220; recovery/oscillation 0.008/250; gradual 1.2/4; confirmed ramp 2/6 to 20/0.5 over 4 minutes; reacquisition Q=6 and R=12 to 0.8.

## TODO in resume order

1. Rerun the full regression after causal isolation; review the 14 legacy snapshot failures without changing expected snapshots automatically.
2. Rerun and review the complete 9-vehicle shadow replay, then regenerate fleet and capacity summaries.
3. Revalidate real traces 29H75028 (2192-2212), 29E45520 persistent downtrend, strong-UP 240-770 and long-U 136-129-137-134 using the isolated dashboard path.
4. Review per-capacity and UNKNOWN behavior; do fleet-level tuning only, never vehicle-specific tuning.
5. Resolve remaining causal ambiguity: a long low plateau without rebound is observationally indistinguishable from a true new baseline until elapsed/stability evidence accumulates.

## Exact resume commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\replay_operational_guard_fleet.py
.\.venv\Scripts\python.exe -m pytest tests\test_operational_guard_golden.py tests\test_dashboard_topic1.py -q
git status --short
git diff --stat
```
