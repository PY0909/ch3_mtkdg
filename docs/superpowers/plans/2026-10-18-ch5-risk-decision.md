# Chapter 5 Risk Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This plan is blocked until the Chapter 4 probabilistic matrix is complete and its per-dataset quality gates are applied.

**Goal:** Consume only frozen, quality-gated Chapter 4 probability artifacts to produce validation-only calibrated risk probabilities and evidence-bounded maintenance decision metrics.

**Architecture:** Treat Chapter 5 as a pure post-processing layer. Load Chapter 4 validation/test distribution artifacts, fit the preregistered calibration family on validation only, derive fixed event thresholds and cost scenarios, and evaluate test risk metrics exactly once. No encoder or probability readout is trained here.

**Tech Stack:** Python 3.9, NumPy, existing artifact schema, pytest.

**Dependency:** `docs/superpowers/plans/2026-10-18-censored-probabilistic-prediction.md`, Tasks 1–9, and the Chapter 4 probability quality gate.

---

## Task 1: Define risk event and calibration contracts

**Files:**
- Create: `refine-logs/CH5_RISK_PREREGISTRATION.md`
- Create: `tests/test_risk_contract.py`

- [ ] Freeze FD event definition `RUL < tau`, the allowed maintenance-horizon values, calibration family, reliability-bin policy, and validation-only fitting procedure.
- [ ] Freeze MetroPT event definition as train-only per-channel p99.5 exceedance, with explicit metadata saying this is an engineering proxy rather than a failure label.
- [ ] Freeze risk threshold policy, risk-coverage construction, lead-time eligibility, and asymmetric cost scenarios.
- [ ] Freeze which Chapter 4 artifacts are eligible: validation probabilities, test probabilities, raw targets, split identity, and quality-gate status.
- [ ] Add tests rejecting missing validation provenance, test-fitted calibrators, unknown event semantics, and artifacts without a Chapter 4 pass record.
- [ ] Run `pytest tests/test_risk_contract.py -q`; expected: all contract tests pass.
- [ ] Commit the preregistration and contract tests before implementing metrics.

## Task 2: Implement validation-only calibration and risk metrics

**Files:**
- Create: `mtkdg/evaluation/risk.py`
- Create: `tests/test_risk_evaluation.py`

- [ ] Implement FD `P(RUL < tau)` extraction from the frozen distribution representation and MetroPT exceedance probability extraction from quantile/distribution artifacts.
- [ ] Implement the preregistered calibration method using validation arrays only; return calibration parameters, fit provenance, and calibrated test probabilities.
- [ ] Implement Brier score, ECE/reliability bins, risk-coverage, hit/miss/false-alarm, and lead time where ordered event sequences support it.
- [ ] Implement expected-cost tables for the fixed early/late/false-alarm ratios; preserve all ratios and do not select one by test performance.
- [ ] Handle empty event bins and zero-event slices explicitly with counts and undefined markers instead of silent zeros.
- [ ] Test split separation, perfect calibration, no-event cases, monotonic risk-coverage, and deterministic cost calculations.
- [ ] Run `pytest tests/test_risk_evaluation.py -q`; expected: all metric and leakage tests pass.
- [ ] Commit the risk evaluator and tests.

## Task 3: Add risk CLI and artifacts

**Files:**
- Create: `scripts/evaluate_risk.py`
- Create: `tests/test_risk_artifacts.py`

- [ ] Accept only a Chapter 4 artifact directory, preregistration ID, and output directory; reject raw model checkpoints or unregistered thresholds.
- [ ] Write calibration parameters, validation/test counts, event semantics, threshold provenance, risk arrays, reliability tables, risk-coverage arrays, lead-time summaries, cost tables, and source artifact hashes.
- [ ] Mark whether the source Chapter 4 dataset quality gate passed and refuse to produce a primary result for failed datasets unless explicitly requested as a diagnostic.
- [ ] Add tests for complete artifacts, failed quality gates, no-event test splits, and repeated invocation.
- [ ] Run `pytest tests/test_risk_artifacts.py -q`; expected: artifact schema and refusal behavior pass.
- [ ] Commit the CLI and artifact tests.

## Task 4: Evaluate and report evidence-bounded Chapter 5 results

**Files:**
- Create: `refine-logs/CH5_RISK_RESULTS.md`
- Create: `scripts/summarize_risk.py`

- [ ] Run the evaluator only on Chapter 4 artifacts whose quality-gate status is present and approved by the preregistration.
- [ ] Summarize all seeds, validation calibration diagnostics, test reliability, risk-coverage, event counts, lead time, and cost tables without selecting a preferred test threshold.
- [ ] Separate FD failure events from MetroPT engineering exceedances in every table and statement.
- [ ] Report Chapter 5 as unsupported for any dataset whose Chapter 4 probability gate fails; retain the failed probability outputs and reasons.
- [ ] Add a final report check for test-once markers, source hashes, and validation-only calibration provenance.
- [ ] Commit the report and summary script as the final Chapter 5 phase.

## Execution constraints

- Do not train models or alter Chapter 4 artifacts in this plan.
- Do not call the test split while fitting calibration, thresholds, or costs.
- Preserve failed datasets and diagnostics; never overwrite historical result directories.
- Keep all changes on `codex/ch4-replan-7aff882`.
