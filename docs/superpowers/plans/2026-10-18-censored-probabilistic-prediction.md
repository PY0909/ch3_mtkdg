# Censored Probabilistic Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Each task ends with focused tests and a phase-scoped commit. Do not launch GPU experiments until Task 2 is complete and its launch gate is accepted.

**Goal:** Add a reproducible frozen-encoder probability workflow for right-censored FD RUL and MetroPT multi-horizon sensor forecasts, with validation-only model selection, one-shot test evaluation, and artifacts for the risk-decision chapter.

**Architecture:** Preserve the third-chapter point runner and add a probability-specific model/runner path that loads a seed-matched frozen encoder and trains only a dataset-specific readout. Shared evaluation functions compute raw-unit censored distribution scores, weighted quantile scores, intervals, and validation-residual baselines; artifacts carry provenance and per-sample predictions for Chapter 5.

**Tech Stack:** Python 3.9, PyTorch 2.5.1, NumPy, pytest, existing `mtkdg` data bundles and checkpoint/artifact conventions.

**Design reference:** `docs/plans/2026-10-18-ch4-censored-probabilistic-prediction-design.md`

---

## File Map

- Create `mtkdg/evaluation/probabilistic.py` for numerically stable distribution losses and scores, interval construction, and validation residual baselines.
- Create `mtkdg/models/probabilistic.py` for frozen-encoder wrappers and dataset-specific readout dispatch; keep shared readout layers from `mtkdg/models/prob_heads.py` only after their contracts are corrected and tested.
- Create `mtkdg/training/probabilistic_runner.py` for readout-only training, validation selection, and one-shot test evaluation.
- Create `scripts/run_probabilistic.py` for explicit dataset, seed, checkpoint, threshold, and artifact arguments.
- Create `scripts/audit_probabilistic_checkpoints.py` to verify all required seed-matched checkpoints and protocol identities before launching the matrix.
- Extend `mtkdg/evaluation/artifacts.py` only for shared provenance helpers if possible; otherwise write probability artifacts through a separate module so point artifacts remain stable.
- Create focused tests under `tests/test_probabilistic_*.py` for numeric functions, model contracts, runner protocol, checkpoint audit, and artifacts.
- Create `refine-logs/CH4_PROBABILISTIC_PREREGISTRATION.md` only after validation metrics, thresholds, splits, calibrators, interval definitions, bootstrap details, and run IDs are frozen and reviewed.
- Preserve all existing point, MC-Graph, and outage artifacts without rewriting them.

## Task 1: Audit frozen encoder checkpoints and protocol identities

**Files:**
- Create: `scripts/audit_probabilistic_checkpoints.py`
- Create: `tests/test_probabilistic_checkpoint_audit.py`

- [ ] Inventory seed 2026/2027/2028 best checkpoints for FD002, FD004, and MetroPT-3 from existing result directories. Read manifests and checkpoint metadata without evaluating test data.
- [ ] Implement a read-only audit that checks existence, SHA256, dataset/split identity, scaler identity, model config, encoder state keys, seed, and validation-best metadata.
- [ ] Add tests for a complete synthetic inventory, missing seed, wrong dataset, wrong split identity, and mismatched encoder keys.
- [ ] Run `pytest tests/test_probabilistic_checkpoint_audit.py -q`; expected: all tests pass and no data/test predictions are loaded.
- [ ] Run the audit against current artifacts and save its machine-readable report under `refine-logs/`.
- [ ] If any checkpoint is missing or incompatible, stop. Do not copy weights or retrain Chapter 3 as part of this task.
- [ ] Commit the audit tool and report as a separate phase commit.

## Task 2: Freeze probability protocol and launch gate

**Files:**
- Create: `refine-logs/CH4_PROBABILISTIC_PREREGISTRATION.md`
- Modify: `docs/plans/2026-10-18-ch4-censored-probabilistic-prediction-design.md` only if a resolved definition needs to be recorded

- [ ] Define the exact FD LogNormal parameterization in raw cycles, likelihood terms for capped and uncapped targets, and stable log-survival calculation.
- [ ] Define the MetroPT quantile levels and weighted interval score formula, including finite-grid endpoint handling; do not label it full CRPS.
- [ ] Define censored validation NLL checkpoint selection for FD and validation WIS selection for MetroPT, patience, epoch budget, and zero-tuning policy.
- [ ] Define FD raw-target test NLL and 50%/90% interval construction, MetroPT aggregate and per-channel/horizon WIS and coverage, and all reported units.
- [ ] Define the validation residual baselines using only validation targets: FD engine-level raw validation residuals and MetroPT channel-by-horizon residuals. Record finite-sample behavior and limitations.
- [ ] Define Chapter 5 calibration families and their validation-only fitting procedure, FD maintenance horizons, MetroPT train-only p99.5 exceedance proxy, bootstrap resampling units, and decision-cost scenarios.
- [ ] Set numerical probability-quality gates before training; express them as predeclared comparison rules against the residual baseline, with no test-driven changes.
- [ ] Verify the checkpoint audit from Task 1 passed for every dataset and seed.
- [ ] Review the preregistration for test leakage and ambiguity; mark the launch gate closed only when all fields are explicit.
- [ ] Commit the preregistration separately. No GPU run is permitted before this commit.

## Task 3: Implement and test stable probability mathematics

**Files:**
- Create: `mtkdg/evaluation/probabilistic.py`
- Modify: `mtkdg/models/prob_heads.py`
- Create: `tests/test_probabilistic_math.py`

- [ ] Write failing tests for LogNormal log-density, right-tail log-survival, censored NLL masking, quantile monotonicity, weighted interval score, central interval extraction, and raw-unit inverse transforms.
- [ ] Test extreme standardized residuals and small/large positive scales; all outputs and gradients must remain finite.
- [ ] Implement stable Normal log-CDF/log-survival primitives using PyTorch special functions or an asymptotic branch with verified gradient behavior; do not use `log(1 - cdf)`.
- [ ] Implement LogNormal density and right-censor likelihood in raw-cycle space, with explicit rejection of nonpositive uncensored targets and correct censoring at `y >= cap`.
- [ ] Correct existing quantile monotonicization so the levels remain ordered and the first quantile is unconstrained; do not use cumulative softplus over every raw output, which forces artificial widening.
- [ ] Implement the preregistered WIS and interval endpoint rules exactly, with shape and finite-value validation.
- [ ] Run `pytest tests/test_probabilistic_math.py -q`; expected: all analytic, gradient, edge-case, and unit tests pass.
- [ ] Commit math and tests as a separate phase commit.

## Task 4: Add frozen encoder probability models

**Files:**
- Create: `mtkdg/models/probabilistic.py`
- Create: `tests/test_probabilistic_models.py`

- [ ] Add a wrapper that builds the same MTK-DG encoder from the audited point checkpoint config, loads only encoder weights, sets `requires_grad=False`, and keeps the encoder in eval mode during readout training.
- [ ] Add the FD LogNormal readout producing location and positive scale from the frozen `[B,C,d]` representation plus condition embedding.
- [ ] Add the MetroPT quantile readout producing `[B,P,C,Q]` with the frozen representation, horizon embedding, and global context.
- [ ] Ensure training forward returns the same tensor contracts used by evaluation; ensure only readout parameters appear in the optimizer.
- [ ] Add tests that compare loaded encoder tensors with the source checkpoint, verify the encoder never changes after optimizer steps, verify output shapes, and exercise multiple seed-specific configs.
- [ ] Run `pytest tests/test_probabilistic_models.py -q`; expected: all shape, freezing, load-compatibility, and gradient-isolation tests pass.
- [ ] Commit model wrappers and tests as a separate phase commit.

## Task 5: Build validation/test probability evaluation and residual baselines

**Files:**
- Modify: `mtkdg/evaluation/probabilistic.py`
- Create: `tests/test_probabilistic_evaluation.py`

- [ ] Implement validation censored NLL for FD using capped labels only; ensure `y_raw_uncapped` never enters the checkpoint-selection calculation.
- [ ] Implement raw-target test NLL, interval coverage/width, and the predeclared `RUL > 125` diagnostic slice.
- [ ] Implement MetroPT validation WIS, test overall and per-channel/per-horizon WIS, coverage, width, and month slices.
- [ ] Implement FD engine-level and MetroPT channel-by-horizon validation residual distribution baselines without reading test targets.
- [ ] Add provenance assertions that each evaluator receives the intended split and that test evaluation can be called only once per run.
- [ ] Test with synthetic known distributions, capped observations, overlapping time windows, and deterministic residual baselines.
- [ ] Run `pytest tests/test_probabilistic_evaluation.py -q`; expected: hand-computed metrics and split guards pass.
- [ ] Commit evaluation functions and tests as a separate phase commit.

## Task 6: Add probability runner, CLI, and artifacts

**Files:**
- Create: `mtkdg/training/probabilistic_runner.py`
- Create: `scripts/run_probabilistic.py`
- Create: `tests/test_probabilistic_runner.py`
- Create: `tests/test_probabilistic_artifacts.py`

- [ ] Add readout-only optimizer setup, training loop, validation-best checkpoint selection, early stopping, and restoration using the frozen encoder contract.
- [ ] Make test evaluation explicit and one-shot; fail if invoked twice or before validation-best restoration.
- [ ] Add CLI flags for dataset, seed, point checkpoint, data root, cap, epochs, patience, device, preregistration ID, and run ID; reject overrides that contradict the preregistration.
- [ ] Save config, code SHA/dirty state, point checkpoint SHA, split/scaler identities, readout version, validation history, test metrics, per-sample predictions/targets, calibration-ready arrays, and an `evaluated_once` marker.
- [ ] Add tests for CLI validation, optimizer parameter membership, best-checkpoint restoration, one-shot test, complete artifacts, and no test files when test evaluation is disabled.
- [ ] Run `pytest tests/test_probabilistic_runner.py tests/test_probabilistic_artifacts.py -q`; expected: all lifecycle and artifact tests pass.
- [ ] Commit runner, CLI, artifact code, and tests as a separate phase commit.

## Task 7: Synthetic end-to-end and CPU data smoke

**Files:**
- Create or modify: `tests/test_probabilistic_smoke.py`

- [ ] Run an end-to-end synthetic FD censored batch and MetroPT quantile batch through forward, loss, backward, checkpoint selection, reload, and artifact writing on CPU.
- [ ] Confirm no encoder parameter changes and only readout gradients are nonzero.
- [ ] Run one CPU data smoke per task using the existing read-only datasets, with test evaluation disabled; verify output shapes, finite validation metrics, and artifact provenance.
- [ ] Run `pytest tests/test_probabilistic_smoke.py -q`; expected: both tasks complete without GPU and do not create test-evaluation artifacts.
- [ ] Commit smoke coverage as a separate phase commit.

## Task 8: Execute the preregistered three-seed matrix

**Files:**
- Create: `scripts/run_probabilistic_matrix.sh`
- Create: `refine-logs/CH4_PROBABILISTIC_RUN_LOG.md`

- [ ] Confirm Task 2 launch gate is committed and all required checkpoints passed Task 1 audit.
- [ ] Run exactly FD002, FD004, and MetroPT-3 × seeds `2026/2027/2028`, using only preregistered configs and unique run directories.
- [ ] Record start/end, command, process exit, checkpoint SHA, run directory, and any failure in the run log; do not silently restart or replace a run.
- [ ] After each run, verify artifact completeness and provenance without using test metrics to alter remaining runs.
- [ ] If a run fails, follow only the preregistered single-retry policy; otherwise retain the missing run and report the actual sample size.
- [ ] Commit the run script and append-only run log after the matrix is complete.

## Task 9: Statistical summary for Chapter 4

**Files:**
- Create: `scripts/summarize_probabilistic.py`
- Create: `refine-logs/CH4_PROBABILISTIC_RESULTS.md`
- Create: `tests/test_probabilistic_summary.py`

- [ ] Aggregate paired per-seed metrics and report all three seed values, mean, and sample standard deviation.
- [ ] Bootstrap FD by engine and MetroPT by natural week within seed; report the exact resampling implementation and confidence interval construction.
- [ ] Compare probabilistic heads with validation-residual baselines and the frozen point model only on compatible metrics.
- [ ] Apply the preregistered probability-quality gates mechanically; record pass/fail by dataset without post-hoc threshold adjustment.
- [ ] Add summary tests using hand-built artifacts for pairing, ddof=1 aggregation, per-seed bootstrap boundaries, and quality-gate application.
- [ ] Run `pytest tests/test_probabilistic_summary.py -q`; expected: deterministic summaries match hand calculations.
- [ ] Run the full test suite once after implementation if runtime is acceptable.
- [ ] Commit summaries and final report artifacts as a distinct reporting phase.

## Execution constraints

- Use `conda activate torch23` for project tests and GPU work; record the environment snapshot in the preregistration.
- Read `/home/new_work/kaf_profiti/dataset` and `/home/cit` only; do not write there.
- Keep every phase commit on `codex/ch4-replan-7aff882`; never merge or cherry-pick into the original `master` without an explicit request.
- Do not run the old MC-Graph matrix or modify its results as part of this plan.
- Do not begin Task 8 until the checkpoint audit and preregistration launch gate pass.
