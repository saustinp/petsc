# Phase 5a — CPU bit-equivalence validation against 129 MATLAB/C++ baselines

**Status:** Plan; ready to execute
**Date drafted:** 2026-05-03
**Predecessors:** Phases 1–4 + 4c all done (#46–#49, #69)
**Baseline source:** `/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/`

---

## 1. TL;DR

Run the PETSc gmstab port against each of 129 baselines, compare its `residuals.csv`
trace against the C++ port's `cpp_residuals.csv`, and produce a Phase-5a
`validation_summary_petsc.csv` mirroring the schema of the existing
MATLAB-vs-C++ `validation_summary.csv`. Multi-tier pass criterion absorbs the
known FP drift between MATLAB and C++ on harder problems. Final deliverable is
a per-baseline pass/drift/fail summary plus aggregate counts comparable to the
89 FAIL_DRIFT / 40 PASS distribution the MATLAB-vs-C++ comparison shows.

---

## 2. Inventory — what's actually in the baseline set

```
/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/
├── asic_320ks/       — N=321,671   nnz=1.32M    Sandia ASIC (SuiteSparse)
├── cdr_small/        — N=729       nnz=4,617    3D FDM CDR, the dev problem
├── cdr_sweep_small/  — 125 sub-baselines, N=729, parameter sweep over (eps, beta_mag, r)
├── ocean/            — N=42,249    nnz=294,795  Stommel ocean stream-function FEM
├── sherman5/         — N=3,312     nnz=20,793   SuiteSparse oil-reservoir
├── torso1/           — N=116,158   nnz=8.52M    SuiteSparse human torso
├── linsys_format.hpp — EXASIMLS format header
├── csr_format_instructions.md
└── validation_summary.csv — MATLAB-vs-C++ status (the existing 129-row reference)
```

**Total: 5 named single baselines + 125 cdr_sweep_small sub-baselines = 129.** ✓

**Critical finding from inventory:** All 129 baselines use **`A_fun = @(v) A*v`** —
unpreconditioned. From `VALIDATION_PACKAGE_SUMMARY.md` §2.3:
> All five [single-point tests] are unpreconditioned by design. The C++ port may
> add preconditioning later; these baselines verify the underlying GM(s)stab
> arithmetic without any ILU dependency mismatching between languages.

This is **intentional** — the package authors deliberately stripped preconditioning
to factor out implementation differences in MATLAB's `\` operator vs C++'s ILU
backends. PETSc has the same parity issue, so the design choice serves us too.

**What this means for Phase 5a's coverage:**

| Configuration | Bit-equivalence (Phase 5a) | Weak correctness (Phase 4) |
|---|---|---|
| **PC_NONE** | ✅ 129 baselines | ✅ implicit |
| **PC_LEFT** + jacobi/bjacobi/ilu/asm | ❌ no baselines exist | ✅ Phase 4b tripwires |
| **PC_RIGHT** + jacobi/bjacobi/ilu/asm | ❌ no baselines exist | ✅ Phase 4a tripwires |
| **PC_SYMMETRIC** + jacobi/bjacobi | ❌ no baselines exist | ✅ Phase 4c tripwires |

Phase 5a validates the **algorithm core** (PC_NONE trajectory matches MATLAB/C++).
It does NOT validate that the preconditioned trajectories match — that's deferred
to **Phase 5b (task #70, follow-up)**, which requires new MATLAB-side baselines
with explicit L/R factor dumps that PETSc can mirror via PCSHELL.

This is a deliberate scope decision, not an oversight. The Phase 4 tripwires give
us solution-correctness for preconditioned modes (the returned `x` satisfies
`||b - A·x|| ≤ tol`); Phase 5b would add trajectory-level bit-equivalence on top of
that. Both layers ultimately want to land, but they require different reference
data and aren't blocking each other.

**Per-baseline files:**
- `linsys.bin` — EXASIMLS format (64-byte header + CSR rowptr/colidx/values + RHS)
- `P.bin` — explicit shadow space (column-major float64)
- `residuals.csv` — MATLAB reference trace
- `cpp_residuals.csv` — C++ port reference trace
- `summary.txt` — problem description, parameters, run results
- Optionally: `cpp_residuals_force_l1.csv`, `cpp_residuals_force_l2.csv`, `cpp_run.log`

**Baseline solver parameters (consistent across all baselines):**
- `s = 4` (shadow dimension)
- `L = 2` (poly stab)
- `tolabs = 1e-10`
- `maxmatvec = 500`
- `maxruntime = 600 s`

---

## 3. Acceptance criterion — multi-tier (NOT bit-equivalence)

**Why not bit-equivalence:** the existing MATLAB-vs-C++ comparison shows
**89/129 are FAIL_DRIFT** (max iter-residual drift > 1e-10) and only **40/129 are
PASS**. The two reference implementations themselves diverge on most problems due
to FP drift in the BGS / polynomial-step / pGMRESm inner solver. Demanding
PETSc-vs-C++ bit-equivalence on problems where MATLAB-vs-C++ already drifts is
unreasonable.

Sam's prior comparison of MATLAB to C++ also surfaced **matvec-count drift up to
~13%** between the two references on harder problems — accepted as a known
consequence of FP-order differences. PETSc adds yet another implementation layer
(its own BLAS/LAPACK call patterns); we should expect similar matvec-count
divergence and budget for it explicitly.

**The right framing for Phase 5a:**
- Don't demand bit-equivalence. Do demand the *destination* of the trajectory matches.
- The two metrics that matter are:
  1. **Final state matches:** convergence reason matches, final residual is in the
     same magnitude band, x is in the same neighborhood
  2. **Matvec count is comparable:** PETSc shouldn't take dramatically more or fewer
     matvecs than the references — that's the actual user-facing perf metric

### Pass tiers (residual-drift axis)

| Tier | Criterion | Indicates |
|---|---|---|
| **PASS** | Max iter-residual drift PETSc-vs-C++ ≤ 1e-10, AND snapshot count exact match | True near-bit-equivalence; algorithm and FP order match closely |
| **PASS_DRIFT** | PETSc-vs-C++ drift bounded by 1.5× MATLAB-vs-C++ drift on the same baseline, AND final residual within 2× of C++'s, AND convergence reason matches (CONVERGED ↔ CONVERGED, DIVERGED ↔ DIVERGED) | Algorithm correct but FP order differs from C++. Acceptable — same drift envelope as MATLAB-vs-C++ |
| **FAIL** | Drift exceeds 1.5× MATLAB-vs-C++ drift OR convergence reason mismatches OR final residual differs by >2× from C++ | Real algorithmic divergence; needs investigation |

### Matvec-count check (independent axis — must also pass)

Beyond residual drift, **PETSc's total matvec count must be within 1.25× of
the references' matvec counts**. This is an *additional* requirement on top of
the residual-drift tier — a baseline can be PASS on residual drift and still
FAIL Phase 5a if its matvec count is out of bounds.

Concretely, given:
- `M_mv` = MATLAB matvec count (from `validation_summary.csv` final_iterres_M's
  corresponding row's `matvec_drift` column gives M_mv − C_mv; M_mv is recoverable)
- `C_mv` = C++ matvec count (read from the last row of `cpp_residuals.csv`)
- `P_mv` = PETSc matvec count (gms->matvec_count after solve)

The matvec gate is:
```
ratio_to_max = P_mv / max(M_mv, C_mv)
ratio_to_min = P_mv / min(M_mv, C_mv)
```

| Matvec result | Criterion |
|---|---|
| **PASS_MV** | `0.8 ≤ ratio_to_min` AND `ratio_to_max ≤ 1.25` (i.e., P_mv is within ±25% of the references) |
| **FAIL_MV** | Outside that band — PETSc converges meaningfully faster or slower than both references |

The 1.25× bound is wider than the ~13% MATLAB-vs-C++ drift Sam observed, on
the principle that adding PETSc as a third implementation layer accumulates
another ~12% headroom (compounding ~13% × 1.12 ≈ 25%).

### Combined verdict

A baseline is **OVERALL_PASS** iff:
- Residual tier ∈ {PASS, PASS_DRIFT}
- Matvec gate = PASS_MV

A baseline is **OVERALL_FAIL** iff:
- Residual tier = FAIL, OR
- Matvec gate = FAIL_MV

The summary CSV records both axes separately so failure mode is diagnosable
("residual drift was fine but matvec count blew up" is a different bug class
from "residual drift was too large but matvec count was fine").

### Special-case bounds for non-converging baselines

For baselines where M_mv = C_mv = maxmatvec (i.e., both references hit the
500-matvec budget without converging, e.g., sherman5, ocean, torso1):
- The matvec ratio is meaningless (everyone is at 500). The matvec gate
  trivially passes if PETSc also hits 500 ± a few.
- Apply only the residual-tier criterion, with the looser "non-converging"
  variant (final residual within 5× of C++, reason matches DIVERGED_ITS).

### Special cases

- **Non-converging baselines** (sherman5, ocean, torso1): Both MATLAB and C++ run to
  matvec budget without reaching tol. Compare PETSc by:
  - Algorithm reason matches DIVERGED_ITS (or equivalent budget-exhausted reason)
  - Final residual within 5× of C++'s final residual (looser bound — since
    MATLAB and C++ already differ by ~3-5× in final residual on these)
- **NaN matches:** if MATLAB or C++ produced NaN at some iter, PETSc must produce
  NaN at a comparable iter (within ±5). The existing `nan_match` column in
  `validation_summary.csv` tracks this for MATLAB-vs-C++; we should track it too.

### Aggregate-level success

Phase 5a is "done" if:

- **PASS + PASS_DRIFT count ≥ MATLAB-vs-C++ PASS count.** I.e., PETSc-vs-C++ should
  agree on at least as many baselines as MATLAB-vs-C++ does (≥ 40 PASS), since
  PETSc is conceptually closer to C++ than MATLAB is.
- **FAIL count ≤ 0**, ideally. Any FAIL is a real bug to chase.
- The full set of 129 baselines runs to completion (no harness crashes, no
  PETSc errors uninterpretable as DIVERGED).

---

## 4. Harness architecture

### Single-binary design

One C binary `tests/ex_gmstab_phase5a_harness.c`:
- Takes `argv[1]` = path to a baseline directory (e.g.,
  `.../baselines/cdr_small`)
- Reads `linsys.bin`, `P.bin`, `summary.txt` (parses tolabs, maxmatvec, etc.
  from the latter)
- Runs PETSc gmstab (PC_NONE, NORM_UNPRECONDITIONED, the natural pairing) and
  emits `petsc_residuals.csv` in the SAME format as `cpp_residuals.csv`
- Loads `cpp_residuals.csv` for comparison
- Prints a one-line summary in the same column schema as `validation_summary.csv`
- Exit code: 0 = PASS or PASS_DRIFT, 1 = FAIL

### Wrapper script

`tests/run_gmstab_phase5a.sh`:
- Iterates every `*/` subdirectory under
  `/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/`
- Calls `ex_gmstab_phase5a_harness` per directory
- Aggregates outputs into `tests/results_phase5a/validation_summary_petsc.csv`
- Generates aggregate counts (PASS / PASS_DRIFT / FAIL) at the end
- Per-failure: emits `results_phase5a/<baseline>/diff_log.txt` with the first
  10 rows where PETSc and C++ diverge

### Why one binary, not 129

- One source file = one place to fix when comparison logic changes
- Keeps the `tests/` directory navigable (vs. 129 near-identical .c files)
- Drives the existing tripwire suite which already follows this pattern (one
  harness binary called repeatedly with different inputs)

---

## 5. Detailed implementation plan

### Step 0 — pre-flight

```bash
[ ] Branch ksp-gmstab, current commit b4515d58ce2 or later
[ ] tripwire suite shows 68/68 (baseline)
[ ] /home/sam/hpc_stack/gmstab_matlab/.../baselines/ is readable
[ ] validation_summary.csv exists and has 130 lines (header + 129)
```

### Step 1 — write the harness binary

`tests/ex_gmstab_phase5a_harness.c`:

```c
/* Phase 5a — single-baseline validator.
   Usage: ex_gmstab_phase5a_harness <baseline_dir> [output_summary_line.csv]
   Reads: <baseline_dir>/linsys.bin
          <baseline_dir>/P.bin
          <baseline_dir>/summary.txt    (parses tolabs, maxmatvec, ||b||)
          <baseline_dir>/cpp_residuals.csv  (reference)
   Writes: <baseline_dir>/petsc_residuals.csv  (this run's trace)
   Prints: a CSV row in validation_summary.csv schema. */
```

Components:

1. **EXASIMLS reader.** Reuse the `load_linsys_parallel` function from existing
   tests (e.g., `ex_gmstab_pcleft_jacobi.c`).

2. **summary.txt parser.** Extract fields with regex:
   - `tolabs`
   - `maxmatvec` (= max_it as iteration count proxy)
   - `||b||_2` (for sanity check)
   The format is consistent across all summary.txt files (verified).

3. **PETSc solve config.** `PC_NONE`, `KSP_NORM_UNPRECONDITIONED`, `s=4`,
   `tolabs` from summary, `maxmatvec` mapped to ksp->max_it.
   Use `-ksp_gmstab_p_file <baseline>/P.bin` to load the explicit shadow space.
   Use `-ksp_gmstab_trace_csv <output_path>` to dump the trace.

4. **CSV diff.** Read `cpp_residuals.csv` and `petsc_residuals.csv` row by row.
   For each row:
   - Compare `iter` and `matvec` (must be exact int match in PASS tier)
   - Compute `iterres_drift = |petsc_iterres - cpp_iterres|`
   - Track `max_iterres_drift`, `first_drift_iter`, `nan_match`, etc.

5. **Tier classification.** Based on the drift envelope MATLAB-vs-C++ shows
   for this baseline (looked up from `validation_summary.csv`).

### Step 2 — write the wrapper script

`tests/run_gmstab_phase5a.sh`:

```bash
#!/usr/bin/env bash
# Phase 5a — sweep all 129 baselines through PETSc gmstab and produce
# validation_summary_petsc.csv.
#
# Usage: ./run_gmstab_phase5a.sh                  # run all 129
#        ./run_gmstab_phase5a.sh --quick          # run small ones first (cdr_sweep_small only)
#        ./run_gmstab_phase5a.sh --filter sandia  # run only matching
#
# Outputs:
#   tests/results_phase5a/
#     validation_summary_petsc.csv   - PETSc-vs-C++ summary (same schema as MATLAB-vs-C++ summary)
#     <baseline>/petsc_residuals.csv - per-baseline PETSc trace
#     <baseline>/diff_log.txt        - failures: first 10 mismatched rows
#     aggregate.txt                  - PASS / PASS_DRIFT / FAIL counts
```

### Step 3 — staged rollout

**Sub-step 3.1 — single-baseline smoke test (cdr_small)**

```bash
./tests/ex_gmstab_phase5a_harness <baselines>/cdr_small/
# Expected: PASS_DRIFT (since MATLAB-vs-C++ is FAIL_DRIFT with drift=0.45 here)
# Inspect the diff_log to confirm divergence is reasonable (similar magnitude
# to MATLAB-vs-C++)
```

This is the canonical dev problem (already used by every Phase-3/4 tripwire).
If gmstab is correct, PETSc-vs-C++ on cdr_small should look similar to
MATLAB-vs-C++.

**Sub-step 3.2 — small problems first (cdr_sweep_small)**

```bash
./tests/run_gmstab_phase5a.sh --filter cdr_sweep_small
# 125 baselines, all N=729. Expected outcome:
#   - 40 PASS / 89 PASS_DRIFT in MATLAB-vs-C++; should match closely in PETSc
#   - Total runtime: ~5-10 minutes (each baseline ~1-3 sec)
```

If many of the existing PASS cases become FAIL in PETSc-vs-C++, we have a real
bug. Use the diff_logs to identify the first divergence iter — likely points
to a specific algorithm step where PETSc's FP order differs from C++.

**Sub-step 3.3 — medium problems**

```bash
./tests/ex_gmstab_phase5a_harness <baselines>/sherman5/
./tests/ex_gmstab_phase5a_harness <baselines>/ocean/
# N=3,312 and N=42,249. Both don't converge (run to matvec budget).
# Expected: PASS_DRIFT on the "didn't converge but stayed bounded" criterion.
```

**Sub-step 3.4 — large problems**

```bash
./tests/ex_gmstab_phase5a_harness <baselines>/torso1/
./tests/ex_gmstab_phase5a_harness <baselines>/asic_320ks/
# N=116,158 and N=321,671.
# torso1 doesn't converge; asic_320ks converges in ~18 matvecs.
# Expected: ~30 sec each.
```

**Sub-step 3.5 — full sweep**

```bash
./tests/run_gmstab_phase5a.sh
```

All 129 in one shot. Expected total runtime: ~15 min serial, less if we
parallelize the wrapper script (each baseline is independent — could use
`xargs -P 4` or similar).

### Step 4 — analyze failures

For any FAIL row in the summary:

1. Read `<baseline>/diff_log.txt` to identify first divergence iter
2. Compare PETSc trace at that iter vs MATLAB and C++ traces — does PETSc
   align with one and not the other? Or with neither?
3. If PETSc aligns with MATLAB but not C++: PETSc inherited a MATLAB-style
   FP order somehow (unlikely; probably indicates a real regression)
4. If PETSc aligns with neither: a third FP order. Investigate the algorithm
   step at that iter (check force_l1/force_l2 traces if present).

Document each failure case in `tests/results_phase5a/failure_analyses/`.

### Step 5 — commit results

The PETSc trace files (`petsc_residuals.csv`) and `validation_summary_petsc.csv`
are not bytes-large but they are the canonical Phase 5a artifact. Commit them
to the repo so future-Claude can compare future PETSc changes against this
baseline.

```bash
git add tests/results_phase5a/
git add tests/ex_gmstab_phase5a_harness.c tests/run_gmstab_phase5a.sh
git commit -m "gmstab: phase 5a — CPU bit-equivalence validation, 129 baselines"
```

---

## 6. Validation gates (Phase 5a is "done" when)

| Gate | Threshold | Rationale |
|---|---|---|
| All 129 baselines run to completion | 0 harness crashes, 0 uninterpretable PETSc errors | Coverage check |
| PETSc PASS + PASS_DRIFT ≥ MATLAB-vs-C++ PASS count | ≥ 40 / 129 | PETSc shouldn't be *worse* than the MATLAB-vs-C++ FP-drift floor on residual drift |
| PETSc OVERALL_PASS on *every* baseline that MATLAB-vs-C++ marks PASS | 40 / 40 | The 40 cleanly-bit-matching cases must remain clean in PETSc on BOTH axes (residual drift + matvec count) |
| OVERALL_FAIL count = 0 | 0 / 129 | Any OVERALL_FAIL is a real bug; do not declare done with outstanding FAILs |
| Final residuals on converging baselines match C++ to within 2× | bounded | Sanity: PETSc converges to comparable accuracy |
| Convergence reason matches C++ on every baseline | exact | CONVERGED/DIVERGED tier match — qualitative outcome |
| **Matvec count within 1.25× of references on every converging baseline** | **0.8 ≤ P_mv / min(M_mv, C_mv) AND P_mv / max(M_mv, C_mv) ≤ 1.25** | **PETSc shouldn't take dramatically more or fewer matvecs. 1.25× is the budget after compounding the known ~13% MATLAB-vs-C++ matvec drift with a similar PETSc-vs-C++ drift envelope** |
| FAIL_MV count = 0 | 0 / 129 (excluding non-converging baselines) | Matvec drift > 25% on a converging baseline indicates an algorithmic or convergence-detection issue worth investigating |

---

## 7. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| PETSc's BLAS/LAPACK call ordering differs from C++ port → mid-trace drift on baselines that PASS in MATLAB-vs-C++ | Medium | PASS_DRIFT tier absorbs reasonable drift; FAIL only on >1.5× MATLAB-vs-C++ drift |
| `summary.txt` parser breaks on unexpected format | Low | All 129 summary.txt are MATLAB-generated with fixed format; verified |
| EXASIMLS reader assumption (idx_size=4, val_size=8) doesn't hold for all baselines | Low | Existing tests verify this for cdr_small; check 1–2 large ones (asic_320ks, torso1) explicitly |
| `maxmatvec` budget hit before convergence (we set max_it = maxmatvec but a single iteration counts as multiple matvecs in IDR) | Medium | Map `maxmatvec → max_it × 5` (each cycle is ~5 matvecs in this algo). Or just set max_it large and trust the trace bounded by maxmatvec |
| Snapshot count differs by 1 due to constructor-row convention | Low | Compare from row 1 onwards (skip constructor); document |
| PETSc gmstab errors on very small or near-singular problems in the sweep | Medium | Catch all errors; classify as FAIL with the error message in diff_log |
| Wrapper script silent-corruption (e.g., wrong baseline dir → wrong comparison) | Low | Each summary line includes baseline name; assert it matches expected |
| Determinism: PETSc's RNG-based shadow-space build doesn't match MATLAB's. We use P.bin (explicit). | None | Already mitigated — every baseline ships P.bin |

---

## 8. Sequencing within Phase 5a

```
Step 0:  Pre-flight (1 min)
Step 1:  Write harness binary (~1 hour)
Step 2:  Write wrapper script (~30 min)
Step 3.1: Smoke test on cdr_small (5 min)
Step 3.2: Sweep cdr_sweep_small (15 min — including any debug iteration)
Step 3.3: Medium problems sherman5, ocean (10 min)
Step 3.4: Large problems torso1, asic_320ks (30 min — runtime-dominated)
Step 3.5: Full sweep (clean run, ~15 min)
Step 4:  Analyze failures, write notes (variable; probably 1-2 hours)
Step 5:  Commit results (5 min)

Total: ~4-5 hours assuming clean runs at each stage.
       Up to 8-10 hours if Step 4 finds substantial divergence requiring algorithmic investigation.
```

---

## 8.5. Diagnostic: detecting accidentally-preconditioned baselines

**Watch-out:** the validation package authors state in `VALIDATION_PACKAGE_SUMMARY.md`
§2.3 that all 129 baselines are unpreconditioned by design. If this is correct,
PETSc-vs-MATLAB/C++ trajectory drift on these problems should be of the order seen
in the existing `validation_summary.csv` (matvec drift up to ~13%, residual drift
typically below 1× max-residual or bounded by the FAIL_DRIFT envelope).

**However**, if any baseline accidentally has preconditioning baked into its
`linsys.bin` matrix (e.g., the matrix stored is `L⁻¹·A·R⁻¹` rather than raw `A`),
PETSc would still operate on it as if it were the original operator. The MATLAB
reference would have generated trajectory data using its corresponding `A_fun`,
which might or might not match the matrix actually written to `linsys.bin`. A
preconditioning mismatch between the dumped matrix and the trajectory data would
cause dramatic divergence — much larger than the ~1.5× drift envelope this plan
budgets for.

**If we see dramatic, anomalous divergence on a baseline that's supposed to be
unpreconditioned, treat it as a flag for investigating whether the baseline is
actually preconditioned**, in addition to the usual debug paths (BLAS-call-order
drift, BGS conditioning, etc.). Specifically:

- Compare `||b||_2` from `summary.txt` against `||b||_2` computed from the
  loaded RHS. Mismatch ⇒ probably preconditioning baked in.
- Compare matrix sparsity pattern against what's described in `summary.txt`
  ("3D FDM CDR, 7-point stencil" should give a diagonal-banded sparsity, not
  a full-fill from a factorization).
- Check that the matrix's diagonal entries are approximately what a
  finite-difference / SuiteSparse problem would have at that stencil/scale.
- If still ambiguous, contact the package authors before chasing PETSc-side
  bug hypotheses.

**Add a sanity-check at harness-load-time** that emits a warning if `||b||`
differs from summary.txt's stated value by more than 1e-12 — cheap to compute,
cheap to log, catches the case where summary metadata and binary data
disagree.

---

## 9. What unblocks after Phase 5a

- **Task #68 (Phase 4 follow-up)** can now execute. The pre-patch bit-equivalence
  baseline is established by this run; re-run after the patch confirms only the
  expected snapshot-skipping rows changed.
- **Phase 4e (GPU PCs)** can start — the CPU baseline is now validated, so any GPU
  divergence is a GPU-port issue, not a base-algorithm issue.
- **Phase 5b (preconditioner trajectory bit-equivalence — task #70)** can start
  once new MATLAB-side baselines are generated with explicit L/R factor dumps
  for split-preconditioning trajectories. Phase 5a's PC_NONE coverage is
  necessary but not sufficient — Phase 5b will close the trajectory gap for
  PC_LEFT, PC_RIGHT, and PC_SYMMETRIC under preconditioned operators.
- **External users** can now run gmstab with confidence on unpreconditioned
  problems and with weak-correctness assurances on preconditioned ones (Phase 4
  tripwires). Phase 5a's `validation_summary_petsc.csv` becomes a publishable
  artifact for the PC_NONE / algorithm-core claim.

---

## 10. Cold-start checklist

If picking this up from no context:

1. Read `USER_MANUAL.md` for solver-level orientation.
2. Read `PHASE3_STATUS.md` for design decisions and audit history.
3. Read this file (`PHASE5A_PLAN.md`) for the validation plan.
4. Read `/home/sam/hpc_stack/gmstab_matlab/.../baselines/cdr_small/summary.txt` for
   the reference-trace format and acceptance criterion.
5. Read `/home/sam/hpc_stack/gmstab_matlab/.../baselines/validation_summary.csv` for
   the existing MATLAB-vs-C++ baseline.
6. Run pre-flight checks (§5 Step 0).
7. Implement Steps 1–2 (harness + wrapper).
8. Execute Steps 3.1 → 3.5 in order.
9. Use Step 4's failure-analysis loop until all baselines pass.
10. Commit per Step 5.

---

## 11. Definition of done

Phase 5a is complete when **all** of the following hold:

- ✅ `tests/ex_gmstab_phase5a_harness.c` exists and compiles cleanly
- ✅ `tests/run_gmstab_phase5a.sh` exists and is executable
- ✅ `tests/results_phase5a/validation_summary_petsc.csv` exists with 129 rows,
  schema: `test, residual_tier, matvec_tier, overall, n_snapshots_P, n_snapshots_C, matvec_count_P, matvec_count_C, matvec_count_M, matvec_ratio_to_max, matvec_ratio_to_min, max_iterres_drift, first_drift_iter, final_iterres_P, final_iterres_C, final_iterres_M, reason_P, reason_C, note`
- ✅ Aggregate counts:
   - PASS + PASS_DRIFT (residual axis) ≥ 40
   - PASS_MV (matvec axis) on every converging baseline
   - **OVERALL_FAIL = 0**
- ✅ Every baseline that MATLAB-vs-C++ marks PASS is also OVERALL_PASS in PETSc
- ✅ All converging baselines (per C++ status) reach final residual within 2× of C++
- ✅ Convergence reason matches C++ on every baseline
- ✅ Matvec count within 1.25× of references on every converging baseline
- ✅ Failure analyses (if any FAILs occurred during dev) documented and resolved
- ✅ Results committed and pushed to `saustinp/petsc:ksp-gmstab`
- ✅ Task #50 marked completed
- ✅ Task #68 noted as unblocked / ready to execute
