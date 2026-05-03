# Phase 5b — Validation against preconditioned MATLAB baselines (375 leaves)

**Status:** Plan drafted; ready to execute
**Date drafted:** 2026-05-03
**Predecessors:** Phases 1–4 + 4c + 5a all complete
**Baseline source:** `/home/sam/hpc_stack/linear_solver_testing/gmstab_precond_pipeline/output/`
**Handoff doc:** `/home/sam/hpc_stack/linear_solver_testing/gmstab_precond_pipeline/handoff.md`

---

## 1. Inventory — verified on disk

| Quantity | Count |
|---|---|
| Test cases (sherman5, torso1, asic_320ks, ocean, cdr_small) | 5 |
| Sweep parameter combinations (cdr_sweep_small) | 120 valid (125 minus 5 SKIPPED) |
| Preconditioning modes per case | 3 (left, right, split) |
| **Total leaves** | **375** ✓ verified by `find` |

Per-leaf files (all 7 verified present in `cdr_small/split/`):

| File | Format | Role |
|---|---|---|
| `linsys.bin` | EXASIMLS header + CSR(A) + RHS(b) | Raw operator and RHS — **mode-independent** |
| `L.bin` | EXASIMLS header + CSR(L) + N zeros (filler) | Lower factor of ILU(0); unit diagonal |
| `U.bin` | EXASIMLS header + CSR(U) + N zeros (filler) | Upper factor of ILU(0) |
| `P.bin` | N×s float64, column-major, no header | Shadow space |
| `b_pre.bin` | N float64, no header | Mode-specific iteration RHS (sanity-check value) |
| `residuals.csv` | 6-col CSV | MATLAB reference trace in mode-native norm |
| `summary.txt` | Human-readable | Operator/RHS/recovery formulas, ILU options, run results, **universal residual** |

Skipped cases (handled gracefully in the harness):
- 1 all-zero combo (ε=β=r=0) — `A=0`, no system to solve
- 4 zero-pivot combos (ε=0, r=0, β∈{1,10,100,1000}) — ILU(0) hits zero pivot

These appear as `SKIPPED` entries in `_workers/worker_*.log`. The harness should detect missing leaves via `os.path.exists(linsys.bin)` and report `SKIPPED_MATLAB` rather than `OVERALL_FAIL`.

---

## 2. The headline caveat — `iterres` is mode-dependent

**This is the single most important thing to get right.** The iteration's internal residual norm is *different* in each mode. The matlab `summary.txt` files re-state this as a per-leaf reminder.

### Operator/RHS/recovery math (verified from `output/README.md` + `cdr_small/right/summary.txt`)

For ILU factors L (unit lower triangular) and U (upper triangular), with `M := L·U`:

| mode | operator A_fun(v) | iter RHS b_pre | iterate→x recovery | iterres column = |
|---|---|---|---|---|
| **none** | `A·v` | `b` | `x = y_iter` | `‖b − A·x_k‖` |
| **left** | `M⁻¹·A·v` = `U⁻¹·(L⁻¹·(A·v))` | `M⁻¹·b` = `U⁻¹·(L⁻¹·b)` | `x = y_iter` | `‖M⁻¹·(b − A·x_k)‖` |
| **right** | `A·(M⁻¹·v)` = `A·(U⁻¹·(L⁻¹·v))` | `b` | `x = M⁻¹·y_iter` = `U⁻¹·(L⁻¹·y_iter)` | `‖b − A·x_k‖` |
| **split** | `L⁻¹·A·U⁻¹·v` | `L⁻¹·b` | `x = U⁻¹·y_iter` | `‖L⁻¹·(b − A·x_k)‖` |

### Mode → PETSc mapping (the natural pairings — no extra work)

| matlab mode | `KSPSetPCSide` | `KSPSetNormType` | gmstab `iter_norm` natively equals |
|---|---|---|---|
| right | `PC_RIGHT` | `KSP_NORM_UNPRECONDITIONED` | `‖b − A·x_user‖` ✓ |
| left | `PC_LEFT` | `KSP_NORM_PRECONDITIONED` | `‖M⁻¹·(b − A·x)‖` ✓ |
| split | `PC_SYMMETRIC` | `KSP_NORM_PRECONDITIONED` | `‖L⁻¹·(b − A·x)‖` ✓ (Phase 4c convention matches) |

**All three are natural pairings.** No extra MatMults beyond the algorithm's native work. ✓

This is also the reason Phase 4c's choice of "PC_SYMMETRIC reports `‖B_L⁻¹·r‖`" was correct — it matches matlab's split-mode convention exactly.

### Sanity: iter=0 iterres values from a sample leaf (cdr_small, ‖b‖=1.16)

| mode | iter=0 iterres | What it represents |
|---|---|---|
| right | 1.16 | `‖b‖` (raw) |
| left | 43.6 | `‖M⁻¹·b‖` (preconditioning amplifies) |
| split | 328.0 | `‖L⁻¹·b‖` (left factor alone amplifies more) |

**⚠️ Constructor-snapshot convention mismatch:** PETSc's gmstab currently emits `iterres = ‖b‖` at the constructor snapshot regardless of mode (`gmstab.c:215`). MATLAB emits the mode-specific norm. The iter=0 row will diverge — **this is a known artifact, not a bug**. The harness must skip iter=0 in the row-by-row diff, or apply a mode-specific iter=0 expected-value override. See §5.5 below.

---

## 3. Architecture

### Single-binary harness with mode arg

```
ex_gmstab_phase5b_harness <baseline_dir> <mode>
```

- `<baseline_dir>` = path to a leaf, e.g. `.../cdr_small/split/`
- `<mode>` = `right`, `left`, or `split`
- Reads `linsys.bin`, `L.bin`, `U.bin`, `P.bin`, `b_pre.bin`, `residuals.csv`, `summary.txt`
- Configures PCSHELL with mode-appropriate apply functions wrapping L and U
- Sets `KSPSetPCSide` + `KSPSetNormType` per the mapping table
- Runs gmstab, dumps trace via `-ksp_gmstab_trace_csv`
- Compares against MATLAB `residuals.csv`
- Computes universal residual `‖b − A·x_recovered‖`
- Emits one CSV row in `validation_summary_petsc_5b.csv` schema
- Exits 0 (OVERALL_PASS), 1 (OVERALL_FAIL), 2 (HARNESS_ERR), 3 (SKIPPED_MATLAB)

Why single binary: same as Phase 5a — one place to fix when comparison logic changes; keeps `tests/` navigable; mirrors the established harness pattern.

### Wrapper script

```
run_gmstab_phase5b.sh
  --filter <substring>    optional, only run matching baselines
  --mode <left|right|split>  optional, only run a specific mode
  --skip-build            skip libpetsc/harness rebuild
```

Iterates all 375 leaves, aggregates into `tests/results_phase5b/validation_summary_petsc_5b.csv`.

---

## 4. The PCSHELL — the novel piece

This is the most critical implementation detail: **a custom PETSc preconditioner that wraps an externally-provided ILU(0) factorization (L and U as separate MATAIJ matrices) and applies the right factor combinations per PCSide.**

### Required PCSHELL ops

```c
struct LUContext {
  Mat L;                  // lower triangular, unit diagonal
  Mat U;                  // upper triangular
  Vec scratch;            // for the two-phase apply (L\v, then U\result)
  PetscInt N;             // for sanity-checking
};

// Apply full M⁻¹: y = U⁻¹ · (L⁻¹ · x)
PetscErrorCode PCApply_ILUFromMatrices(PC pc, Vec x, Vec y);

// Apply only L⁻¹ (left factor) — used by PC_SYMMETRIC's bLocal preconditioning
PetscErrorCode PCApplySymmetricLeft_ILUFromMatrices(PC pc, Vec x, Vec y);

// Apply only U⁻¹ (right factor) — used by PC_SYMMETRIC's unwrap
PetscErrorCode PCApplySymmetricRight_ILUFromMatrices(PC pc, Vec x, Vec y);

// Setup: nothing to do (L and U are already factored)
PetscErrorCode PCSetUp_ILUFromMatrices(PC pc);

// Destroy: free the context (L, U, scratch are owned by harness, not pc)
PetscErrorCode PCDestroy_ILUFromMatrices(PC pc);
```

### Triangular solve implementation — custom CSR-based

PETSc has `MatForwardSolve` / `MatBackwardSolve` but they only work on factor-typed matrices, not regular MATAIJ. Two options:

**Option A (chosen):** custom forward/backward solve directly on CSR data.

```c
// Forward solve L y = x, L is lower triangular with unit diagonal.
// L is MATSEQAIJ; we read its CSR via MatSeqAIJGetIJ + MatSeqAIJGetArray.
//
// Algorithm (sequential, row-by-row):
//   for i = 0..N-1:
//     y[i] = x[i]
//     for each (j, val) in row i of L with j < i:
//       y[i] -= val * y[j]
//     // unit diagonal: skip divide by L[i,i]=1
//
static PetscErrorCode TriangularSolve_Forward(Mat L, Vec x, Vec y) {
  const PetscInt    *ai, *aj;
  const PetscScalar *av;
  PetscInt           N, i, k;
  PetscScalar        sum;
  PetscScalar       *y_arr;
  const PetscScalar *x_arr;

  PetscFunctionBegin;
  PetscCall(MatGetSize(L, &N, NULL));
  PetscCall(MatSeqAIJGetCSRAndMemType(L, &ai, &aj, &av, NULL));
  PetscCall(VecGetArrayRead(x, &x_arr));
  PetscCall(VecGetArray(y, &y_arr));
  for (i = 0; i < N; ++i) {
    sum = x_arr[i];
    for (k = ai[i]; k < ai[i+1]; ++k) {
      const PetscInt j = aj[k];
      if (j < i) sum -= av[k] * y_arr[j];
      // j == i is unit diagonal (skip); j > i shouldn't happen for L
    }
    y_arr[i] = sum;  // unit diagonal so no divide
  }
  PetscCall(VecRestoreArrayRead(x, &x_arr));
  PetscCall(VecRestoreArray(y, &y_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Backward solve U y = x:
//   for i = N-1..0:
//     diag = U[i,i]
//     y[i] = (x[i] - sum_{j>i} U[i,j] * y[j]) / diag
static PetscErrorCode TriangularSolve_Backward(Mat U, Vec x, Vec y) {
  // Symmetric to Forward, but iterates from N-1 down and divides by diag.
}
```

Pattern-F (latent uninit) note: every `y[i]` is written exactly once; no read-before-write hazard.

**Option B (rejected):** use `MatSetFactorType(M, MAT_FACTOR_LU)` and `MatSolve`. Requires combining L and U into a single MATSEQAIJ with the factor flag, which is internal-API territory. Custom solve is simpler and just as fast for our problem sizes.

### Per-mode PCSHELL op selection

```c
PetscCall(PCShellSetApply(pc, PCApply_ILUFromMatrices));
if (mode == MODE_SPLIT) {
  // PC_SYMMETRIC needs both half-applies; ensure we set them up
  PetscCall(PCShellSetApplySymmetricLeft(pc, PCApplySymmetricLeft_ILUFromMatrices));
  PetscCall(PCShellSetApplySymmetricRight(pc, PCApplySymmetricRight_ILUFromMatrices));
}
```

For `right` and `left` modes, only `PCApply` is exercised (full M⁻¹). For `split`, all three ops are used.

### Sanity check at PCSHELL setup

After loading L, U, and b_pre.bin, before calling `KSPSolve`:
1. Apply our PCApply (or PCApplySymmetricLeft for split) to b
2. Compare result against b_pre.bin
3. If `‖our_b_pre − matlab_b_pre‖_2 > 1e-12`, **abort with HARNESS_ERR** — the triangular solve is broken or the L/U/b values disagree with what matlab used

This catches Pattern-D (snapshot rhythm divergence) at the input boundary: if our PCApply produces a different b_pre than matlab's, we'd silently drift from iter 0 and the comparison would fail uninformatively.

---

## 5. Harness binary detail (file layout)

### 5.1 Reads

```
linsys.bin    -> Mat A, Vec b              (load_linsys, reused from Phase 5a)
L.bin         -> Mat L                      (extract CSR from EXASIMLS, build MATSEQAIJ)
U.bin         -> Mat U                      (same)
P.bin         -> set via -ksp_gmstab_p_file (existing path)
b_pre.bin     -> Vec b_pre_matlab           (raw float64 read; sanity-check value)
residuals.csv -> Trace cpp_trace            (load_trace, reused from Phase 5a)
summary.txt   -> BaselineMeta (+ universal_residual_ref) (parse_summary_5b)
```

### 5.2 KSP configuration per mode

```c
KSP ksp;
PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
PetscCall(KSPSetOperators(ksp, A, A));
PetscCall(KSPSetType(ksp, KSPGMSTAB));
PetscCall(KSPSetTolerances(ksp, 0.0, meta.tolabs, PETSC_DEFAULT, petsc_max_it));

PC pc;
PetscCall(KSPGetPC(ksp, &pc));
PetscCall(PCSetType(pc, PCSHELL));
PetscCall(PCShellSetContext(pc, &lu_ctx));
PetscCall(PCShellSetApply(pc, PCApply_ILUFromMatrices));

switch (mode) {
case MODE_RIGHT:
  PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  break;
case MODE_LEFT:
  PetscCall(KSPSetPCSide(ksp, PC_LEFT));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
  break;
case MODE_SPLIT:
  PetscCall(KSPSetPCSide(ksp, PC_SYMMETRIC));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
  PetscCall(PCShellSetApplySymmetricLeft(pc, PCApplySymmetricLeft_ILUFromMatrices));
  PetscCall(PCShellSetApplySymmetricRight(pc, PCApplySymmetricRight_ILUFromMatrices));
  break;
}
PetscCall(KSPSetFromOptions(ksp));  // honor -ksp_gmstab_s, -ksp_gmstab_p_file, -ksp_gmstab_trace_csv
```

### 5.3 max_it derivation

Same as Phase 5a — derive from C++ trace's snap-per-matvec ratio:

```c
int    cpp_final_matvec = cpp_trace.rows[cpp_trace.n - 1].matvec;
double snap_per_mv      = (double)cpp_trace.n / (double)cpp_final_matvec;
int    petsc_max_it     = (int)ceil(meta.maxmatvec * snap_per_mv) + 5;
```

(But Phase 5b uses MATLAB residuals.csv, not C++. So compute from matlab_trace instead.)

### 5.4 Comparison logic (mode-agnostic — the trace columns are already mode-native)

Because PETSc and MATLAB both report iterres in the same mode-native norm, the row-by-row drift comparison is straightforward:

```c
for (i = 1; i < min(petsc_trace.n, matlab_trace.n); ++i) {  // skip iter=0
  drift_iterres = fabs(petsc_trace.rows[i].iterres - matlab_trace.rows[i].iterres);
  if (drift_iterres > max_drift) {
    max_drift = drift_iterres;
    first_drift_iter = i;
  }
}
```

**Critical: skip iter=0** because of the constructor-snapshot convention mismatch (§2 footnote). Document this loudly in the harness comments.

### 5.5 Universal residual cross-check (the strongest correctness test)

After `KSPSolve`, recover the user-visible solution `x_user` per the mode's recovery formula:

```c
Vec x_recovered;
VecDuplicate(b, &x_recovered);
switch (mode) {
case MODE_LEFT:
case MODE_RIGHT:
  // PETSc's PC_LEFT/PC_RIGHT FinalizeSolution_Private already produced
  // x_user in vec_sol. So x_recovered = vec_sol directly.
  VecCopy(x_petsc, x_recovered);
  break;
case MODE_SPLIT:
  // PC_SYMMETRIC's FinalizeSolution applies PCApplySymmetricRight = U⁻¹.
  // So x_user = U⁻¹·y_iter, which is what vec_sol already contains.
  // (Phase 4c verified this.)
  VecCopy(x_petsc, x_recovered);
  break;
}
// Universal residual: ||b - A·x_recovered||
Vec Ax, r;
VecDuplicate(b, &Ax); VecDuplicate(b, &r);
MatMult(A, x_recovered, Ax);
VecWAXPY(r, -1.0, Ax, b);
VecNorm(r, NORM_2, &petsc_universal_residual);
```

Compare `petsc_universal_residual` to:
- The `‖b − A·recover_fn(x_iter)‖_2` value parsed from summary.txt — should agree to FP precision (e.g., within `100 * tolabs`)
- The same value parsed from the OTHER two modes' summary.txt files for this case — should also agree (cross-mode consistency)

This is the **strongest sanity check available**: regardless of mode, regardless of trajectory drift, all three modes for the same case should produce x_user values that all satisfy `‖b − A·x_user‖ ≈ same number`.

### 5.6 Output schema

`validation_summary_petsc_5b.csv` schema:

```
case, mode, residual_tier, matvec_tier, overall, n_snap_P, n_snap_M, matvec_count_P, matvec_count_M,
max_iterres_drift_PM, first_drift_iter, final_iterres_P, final_iterres_M, reason_P, matlab_converged,
universal_res_P, universal_res_summary, universal_res_drift, bnorm_actual, bnorm_summary, bnorm_warn,
b_pre_drift, note
```

New columns vs. Phase 5a:
- `mode` — left/right/split
- `universal_res_P`, `universal_res_summary`, `universal_res_drift` — the cross-mode sanity check
- `b_pre_drift` — sanity check at PCSHELL setup (our PCApply(b) vs matlab's b_pre.bin)

---

## 6. Acceptance criteria

### 6.1 Per-leaf classification (multi-tier, refined from Phase 5a)

**Residual tier** (drift between PETSc and MATLAB iterres columns):

| Tier | Criterion |
|---|---|
| PASS | max iter≥1 drift `≤ 1e-10` (the matlab handoff's stated target) |
| PASS_DRIFT | PETSc final within 10× of MATLAB final (with tolabs floor); reason matches; universal residual within 1e-6 of summary.txt |
| FAIL | outside that envelope |

**Matvec tier:**

| Tier | Criterion |
|---|---|
| PASS_MV | PETSc matvec count within 0.8x..1.25× of MATLAB's (handoff says "identical when tolabs binds, ±3 when maxmatvec binds" — we use the looser 25% from Phase 5a as the practical envelope) |
| FAIL_MV | outside |
| NA_MV | matlab didn't converge — matvec ratio meaningless |

**b_pre sanity:** harness aborts with HARNESS_ERR if `‖our_PCApply(b) − matlab_b_pre‖ > 1e-12`. This isn't a tier — it's a hard precondition for the comparison to be meaningful at all.

**Universal residual sanity:** if the universal residual disagrees with summary.txt by more than `1e-6`, **flag in note**. Doesn't auto-fail (could be FP noise on hard-stagnation cases) but warrants investigation.

**OVERALL_PASS:** `residual_tier ∈ {PASS, PASS_DRIFT}` AND `matvec_tier ∈ {PASS_MV, NA_MV}` AND `b_pre_drift ≤ 1e-12` AND universal_res sanity OK (or noted).

### 6.2 Aggregate gates

Phase 5b is "done" when:

| Gate | Threshold |
|---|---|
| All 375 leaves run to completion | 0 HARNESS_ERR (other than expected SKIPPED_MATLAB) |
| OVERALL_PASS count | All non-skipped baselines |
| OVERALL_FAIL count | 0 |
| Universal residual cross-mode consistency | Each case's three modes produce universal residuals within 1e-6 of each other |
| b_pre sanity | 0 leaves with `b_pre_drift > 1e-12` |
| Existing tripwire suite | 68/68 unchanged (no regression) |

### 6.3 What "PASS" looks like for each mode

- **Right mode:** PETSc-vs-MATLAB drift should be very tight, matching Phase 5a's PC_NONE pattern. Expect ~30-40% PASS, rest PASS_DRIFT — same FP-order divergence story.
- **Left mode:** Similar drift pattern. The `M⁻¹` application is exactly the same operation in both PETSc and MATLAB (same triangular solves on same L, U), so the iterres values should track tightly.
- **Split mode:** Exercises PC_SYMMETRIC + PCApplySymmetricLeft + PCApplySymmetricRight. Critical for validating Phase 4c's correctness. Strong test.

---

## 7. Implementation steps

### Step 0 — pre-flight

```
[ ] Branch ksp-gmstab, current commit at 9840af970ec or later
[ ] Tripwire suite: 68/68 (baseline)
[ ] /home/sam/hpc_stack/linear_solver_testing/gmstab_precond_pipeline/output/ readable
[ ] 375 leaves under output/baselines/
```

### Step 1 — write the L/U-from-EXASIMLS reader

Helper function `load_lu_matrix(path) -> Mat`. Reads EXASIMLS header (same format as linsys.bin), allocates MATSEQAIJ, fills with the CSR data, ignores the trailing N-zero "filler RHS" section. Reuse the load_linsys core but with no Vec b output.

### Step 2 — write the PCSHELL ILU wrapper

`include/pcshell_ilu_from_matrices.h` (private header in tests/) declares:
- `LUContext` struct
- `PCApply_ILUFromMatrices`, `PCApplySymmetricLeft_ILUFromMatrices`, `PCApplySymmetricRight_ILUFromMatrices`
- Helper `TriangularSolve_Forward`, `TriangularSolve_Backward`
- Setup helper that takes a `PC pc` and an `LUContext *ctx` and wires it all up

### Step 3 — write the harness binary

`tests/ex_gmstab_phase5b_harness.c`. Mirror `ex_gmstab_phase5a_harness.c` structure but extended for the precond case:
- Args: baseline_dir, mode
- Read all 7 leaf files
- Set up Mat L, Mat U, Vec b_pre_matlab
- Sanity-check PCSHELL by applying it to b and diffing against b_pre_matlab
- Configure KSP per mode mapping
- Solve, dump trace, compare row-by-row (skipping iter=0)
- Compute universal residual, compare to summary.txt and across modes
- Emit summary line

### Step 4 — write the wrapper script

`tests/run_gmstab_phase5b.sh`. Iterate all 375 leaves. Aggregate.

### Step 5 — staged rollout

**Sub-step 5.1:** smoke test on `cdr_small/right` (the canonical / fastest path)
- Right mode = PETSc PC_RIGHT + NORM_UNPRECONDITIONED, the natural "free" pairing
- Expected: matches Phase 4a's pcright_jacobi behavior, OVERALL_PASS

**Sub-step 5.2:** smoke test on `cdr_small/left`
- Left mode = PETSc PC_LEFT + NORM_PRECONDITIONED, the natural pairing
- Expected: OVERALL_PASS, drift tight (no Phase 4b stall — natural pairing is correct)

**Sub-step 5.3:** smoke test on `cdr_small/split`
- Split mode = PETSc PC_SYMMETRIC + NORM_PRECONDITIONED — exercises Phase 4c
- Sanity-check: PCApplySymmetricLeft(b) should match b_pre.bin to FP precision (since matlab's split-mode b_pre = L⁻¹·b)
- Expected: OVERALL_PASS, validates Phase 4c implementation against MATLAB ground truth

**Sub-step 5.4:** all 5 named cases × 3 modes = 15 leaves
- Run each, compare. Spot-check failures.

**Sub-step 5.5:** cdr_sweep_small (120 × 3 = 360 leaves)
- Run via wrapper. Expected runtime: ~3-5 minutes serial.

**Sub-step 5.6:** full sweep (375 leaves)
- Wrapper end-to-end. Expected runtime: ~5-10 minutes.

### Step 6 — cross-mode consistency check

For each case, compare the universal residual values across its three modes:

```python
for case in CASES:
    u_left  = read_summary_universal_res(f"{case}/left/summary.txt")
    u_right = read_summary_universal_res(f"{case}/right/summary.txt")
    u_split = read_summary_universal_res(f"{case}/split/summary.txt")
    # All three should agree to FP precision (within 100*tolabs typically)
    assert max(u_left, u_right, u_split) - min(u_left, u_right, u_split) < 1e-6
    # And PETSc's universal residuals from each mode should also agree
```

This is the strongest test — independent of trajectory, independent of mode, the user-visible answer must be the same.

### Step 7 — failure analysis

For any FAIL row:
1. Read the per-leaf log
2. Inspect `b_pre_drift` — is the PCSHELL applying L/U correctly?
3. Inspect first_drift_iter — does it correlate with the Phase 4 PC-side dispatch sites?
4. Compare to validation_summary_petsc.csv from Phase 5a (for the same matrix, no-precond) — is the underlying algorithm working?

### Step 8 — commit

Files committed:
- `tests/ex_gmstab_phase5b_harness.c`
- `tests/pcshell_ilu_from_matrices.h` (helper header) and `.c` (or merged into harness)
- `tests/run_gmstab_phase5b.sh`
- `tests/results_phase5b/validation_summary_petsc_5b.csv`
- `tests/results_phase5b/aggregate.txt`
- `tests/results_phase5b/.gitignore` (skip per-leaf logs)
- Doc updates to `USER_MANUAL.md` (note: PCSHELL pattern for matlab L/U handoff is in test code if users want to mirror it)
- Doc updates to `PHASE5B_PLAN.md` (this file) — mark "Status: complete"

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Custom triangular solve has a bug → all leaves fail with `b_pre_drift > 1e-12` | Medium | b_pre sanity check catches it at HARNESS setup time — won't silently corrupt comparisons. Also can validate triangular solve against PETSc's `PCApply(PCILU built on A)` on a known SPD case |
| iter=0 row diverges and we forget to skip → all leaves report FAIL | Low | Documented skip-iter=0 logic; verify on first cdr_small/left smoke test |
| matvec_drift on edge-case sweep baselines | Medium | Same mitigation as Phase 5a (10× envelope, NA_MV for non-converging) |
| PETSc PCSHELL with applysymmetricleft/right has an undocumented gotcha for split mode | Low | Validated empirically by Phase 4c's `ex_gmstab_pcsymmetric_sweep` test that uses PCJACOBI's symmetric apply. Same plumbing used here |
| Constructor-snapshot mode-mismatch is more than just iter=0 | Low | If iter=1 also diverges meaningfully (>>1e-10), the algorithm itself diverges; investigate |
| Universal residual disagrees with summary.txt | Medium | Could be: (a) PETSc's recovery doesn't match matlab's recover_fn, (b) FP noise on stagnation cases. Distinguish via the cross-mode-consistency check |
| L/U files have unexpected sparsity pattern (non-strict-lower, non-strict-upper) | Low | Matlab handoff guarantees `signature_ok` and `pattern_ok`; sanity-check after load: assert `aj[k] <= i` for all k in L's row i, and `aj[k] >= i` for all k in U's row i. Bail with HARNESS_ERR if violated |
| Triangular solve fails on a singular row (`U[i,i] == 0`) | Low | Matlab pre-validated; the 4 SKIPPED zero-pivot baselines are the only cases this could happen, and they're filtered out at the directory-existence check |

---

## 9. Pattern-A through Pattern-F audit checklist (post-implementation)

After Step 5.6 completes, verify the harness code:

| Pattern | What to check |
|---|---|
| **A** Init-once-never-reset | Each invocation allocates fresh `Mat L, Mat U, Vec scratch, LUContext`. No global state across baselines |
| **B** Multi-path invariant | Single OVERALL_PASS gate; mode dispatch happens once at config time, not interleaved with comparison |
| **C** Lazy-init dispatch | PCSHELL ops are set explicitly per mode at config time; no runtime branching in the inner solve |
| **D** Snapshot rhythm divergence | The harness's `compare_traces` compares iter-aligned rows from row 1 onwards; iter=0 deliberately excluded with comment |
| **E** Off-by-statement bookkeeping | `b_pre` sanity check fires BEFORE KSPSolve; trace CSV emission happens during the solve; comparison happens AFTER |
| **F** Latent uninit state | `LUContext` zero-initialized via `{0}` then explicitly populated; triangular solve writes y[i] exactly once per i |

---

## 10. Cold-start checklist

If picking this up after a context loss:

1. Read `handoff.md` for the matlab-side overview
2. Read `output/README.md` for the iterres-norm-per-mode caveat (the single most important thing)
3. Read one `summary.txt` (e.g., `cdr_small/split/summary.txt`) to see the per-leaf format
4. Read this file (`PHASE5B_PLAN.md`) for the implementation blueprint
5. Verify pre-flight checks (§7 Step 0)
6. Implement Steps 1–4 (helpers + harness + wrapper)
7. Execute staged rollout 5.1 → 5.6
8. Use Step 7 (failure analysis) loop until aggregate gates met
9. Commit per Step 8

---

## 11. Definition of done

Phase 5b is complete when **all** of the following hold:

- ✅ Harness binary `ex_gmstab_phase5b_harness.c` compiles cleanly
- ✅ Wrapper script `run_gmstab_phase5b.sh` runs end-to-end
- ✅ `validation_summary_petsc_5b.csv` exists with 375 rows (or 371 if 4 SKIPPED_MATLAB are excluded)
- ✅ Aggregate counts: OVERALL_PASS = all non-skipped, OVERALL_FAIL = 0
- ✅ Every leaf's b_pre sanity check passes (drift ≤ 1e-12)
- ✅ Universal residual cross-mode consistency holds for all 5 cases × 3 modes (the strongest test)
- ✅ Existing 68/68 tripwire suite still green (no regression)
- ✅ Pattern-A through Pattern-F audit clean
- ✅ Phase 4c (PC_SYMMETRIC) effectively validated against MATLAB ground truth via the split-mode column
- ✅ Results committed and pushed to `saustinp/petsc:ksp-gmstab`
- ✅ Task #70 marked completed

---

## 12. What this validates that Phase 5a didn't

| Capability | Phase 5a (no precond) | Phase 5b (precond) |
|---|---|---|
| Algorithm core arithmetic | ✓ 129/129 | ✓ (re-verified across 375 leaves) |
| `PC_NONE` dispatch | ✓ | (out of scope; not retested) |
| `PC_LEFT` dispatch trajectory bit-equivalence | ✗ (no baselines) | ✓ via 125 left-mode leaves |
| `PC_RIGHT` dispatch trajectory bit-equivalence | ✗ (no baselines) | ✓ via 125 right-mode leaves |
| `PC_SYMMETRIC` dispatch trajectory bit-equivalence | ✗ (no baselines) | ✓ via 125 split-mode leaves — **closes the Phase 4c gap** |
| PCApplySymmetricLeft / PCApplySymmetricRight numerical correctness | indirect (via Phase 4c sweep) | direct (matlab L,U identical to PETSc's) |
| Universal residual cross-mode consistency | n/a | ✓ — can independently verify each case's three modes produce same x_user |
| User-side ILU(0) integration via PCSHELL | n/a | ✓ — establishes the canonical pattern for users wrapping their own L/U |

Phase 5b is the stronger test in every dimension that involves preconditioning. Combined with Phase 5a, the gmstab port is then validated across the full (algorithm × PC-side × norm-type × matrix-source) cross-product.

---

## 13. Estimated effort

| Step | Effort |
|---|---|
| Step 1 (L/U reader) | ~30 min |
| Step 2 (PCSHELL + triangular solves) | ~1.5 hours |
| Step 3 (harness binary) | ~1.5 hours |
| Step 4 (wrapper script) | ~30 min |
| Step 5 (staged rollout, including likely 1-2 debug iterations) | ~1.5 hours |
| Step 6 (cross-mode consistency check + plotting if useful) | ~30 min |
| Step 7 (failure analysis, if any FAILs) | ~1 hour (variable) |
| Step 8 (commit, doc updates) | ~30 min |
| **Total** | **~6-8 hours** clean run |

Phase 5b is a meatier implementation than Phase 5a because of the PCSHELL infrastructure, but the comparison logic reuses Phase 5a's tier classification. The main risk-bearing piece is the triangular solve correctness — once `b_pre_drift ≤ 1e-12` clears on the first leaf, the rest should follow the Phase 5a pattern.
