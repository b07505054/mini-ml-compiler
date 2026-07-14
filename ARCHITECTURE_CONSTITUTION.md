# Architecture Constitution

Last verified: 2026-07-14.

Canonical compiler HEAD: `ba388a93eeccd11045f8c1f2eb950ede2601bc88`.
Canonical runtime HEAD: `53c80e2c11101ec7b8db2e73f978e220c054d9a1`.
Canonical capabilities HEAD: `aac593da0bdde7a95c38c03920fc4d00b73011db`.

## Identity

The project is an IR-centered, hardware-aware, evidence-driven implementation-decision compiler for Edge AI backends.

It may use external runtimes and kernel libraries as implementation candidates. Its central value is not reimplementing every kernel; it is choosing, validating, and materializing implementation decisions with explicit provenance.

## Canonical Pipeline

```text
Model / GenericGraphIR
  -> Semantic IR
  -> Program Analysis
  -> Candidate Providers
  -> complete ImplementationCandidates
  -> Feasibility
  -> Evidence-backed Policy
  -> PolicyResult
  -> Implementation IR or Execution Contract
  -> Runtime validates
  -> Runtime executes exactly
  -> Real measurement
  -> Offline calibration
```

## Core Rule

**Compiler chooses. Runtime validates. Runtime executes.**

Canonical execution paths obey this rule. Historical toy planners, benchmark scripts, simulators, and non-canonical evaluation utilities may exist, but they must not be documented as compiler authority unless they invoke the live Compiler and consume its emitted decision artifact.

Evaluation harnesses may not duplicate compiler decisions. E2.1 violated this rule by hardcoding the project threshold and kernel ID in Python. E3 repaired it by invoking the live Compiler and consuming a compiler-generated comparison contract.

## Layer Authority

Semantic IR owns what the program computes: operator semantics, tensor ranks, shapes, dtypes, constants, regions, dependencies, and model structure.

Candidate Providers enumerate legal implementation options at an explicit scope. Providers do not select, benchmark online, parse result files, or own global policy.

Feasibility decides whether a candidate is legal and usable for the target, artifact, shape, dtype, runtime, input binding, and provenance constraints. Evidence cannot legalize an invalid artifact.

Evidence records measured, predicted, declared, oracle, regret, correctness, accuracy, provenance, and truth-boundary facts. Evidence can rank feasible candidates only after legality.

Policy selects among feasible candidates under an objective and records the selection in `PolicyResult`.

Materialization derives Implementation IR or an Execution Contract from the selected candidate. Opaque external libraries may remain as exact contracts when the compiler does not own internal lowering.

Runtime validates exact identities and executes. Runtime may fail explicitly. Runtime fallback is allowed only when the compiler contract explicitly permits it.

## Runtime Non-Goals

Runtime must not search backends, search kernels, search tiles, search thread schedules, choose precision, benchmark online for implementation selection, silently fallback, or silently rewrite compiler decisions.

## Truth Boundaries

Every current claim must be classified as one or more of: production/canonical, measured, calibrated, derived, predicted, declared-profile, rule-based, shadow, experimental, historical, invalid, planned, or missing.

Not every compiler decision is measured-profile-driven. P1D.1 and E3 demonstrate closed evidence loops. Other paths remain declared-profile, rule-based, shadow, or experimental.
