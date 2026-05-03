# Phase 4 plan — preconditioner sides for `KSPGMSTAB`

**Status:** drafted 2026-05-03 after Phase 3d closed at 40/40 tripwires.

## Goals

Bring `KSPGMSTAB` to functional parity with PETSc's standard expectation
that every `KSPType` works with the full PC framework. Concretely:

- Every supported `(pc_side, normtype)` pair declared via
  `KSPSetSupportedNorm` in `KSPCreate_GMSTAB` actually solves correctly
  — i.e. the user-visible `x` in `ksp->vec_sol` satisfies
  `‖b − A·x‖ ≤ tolabs` (or whatever the user's tolerance is).
- Bit-equivalence with the C++ reference is preserved on the
  `(PC_NONE, KSP_NORM_UNPRECONDITIONED)` path that Phase 3d validated;
  introducing PC handling does not regress that path.
- Standard PETSc preconditioners (`PCJACOBI`, `PCBJACOBI`, `PCILU`,
  `PCSOR`, `PCASM`, `PCGAMG`, `PCHYPRE`) all produce convergent solves
  on `cdr_small` regardless of `pc_side` (sequential first, then
  parallel).

## Current state (Phase 3 result)

| Path | Status |
|---|---|
| `(PC_NONE, KSP_NORM_UNPRECONDITIONED)` sequential + parallel | bit-equivalent with C++ at 1.31e-12 (cycle1) / 5.26e-12 (cycle2); structural correctness on natural-flow |
| `(PC_RIGHT, *)` | UNTESTED — likely broken: solution unwrap missing |
| `(PC_LEFT, *)` | UNTESTED — likely broken: bLocal not in preconditioned algebra |
| `(PC_SYMMETRIC, *)` | UNTESTED — algebra not even enumerated |

The algorithm uses `KSP_PCApplyBAorAB` for every Krylov matvec, so the
*dispatch* is correct in all PC modes. What's broken is what we do
*around* the dispatch: initial residual computation, solution
representation, and snapshot trueres recomputation.

## The algebra — what each PC side requires

### PC_NONE (validated in Phase 3)
- Operator: `M = A`, residual `r = b − A·x`.
- `x_user = x_global + x_local` (current code).

### PC_RIGHT
- Mathematical setup: solve `A·B⁻¹·y = b` where `y = B·x_user`.
- `KSP_PCApplyBAorAB` returns `A·B⁻¹·in` — i.e. the algorithm sees
  `M = A·B⁻¹` when it does a matvec.
- Algorithm-internal "x" is the M-space variable. Define
  `x_alg = x_global + x_local` (sum of all xa contributions across
  restarts plus the current cycle's xa).
- Relationship to user: `x_user = x_initial + B⁻¹ · x_alg`. (Where
  `x_initial` is the user's initial guess at solve start.) The
  algorithm's xa accumulates corrections in M-space; `B⁻¹` maps them
  back to x_user space.
- **Required at solve end:** apply `B⁻¹` to `(x_global + x_local −
  x_initial)`, then add `x_initial` back, to recover `x_user`. We
  currently do `x_local += x_global` only, which gives `x_alg`, not
  `x_user`.
- **Required in `Snapshot_Private`'s trueres computation:** unwrap
  `x_total → x_user` before `MatMult(A, x_user)`, otherwise
  `‖b − A·x_total‖ ≠ ‖b − A·x_user‖`.

### PC_LEFT
- Mathematical setup: solve `B⁻¹·A·x = B⁻¹·b`.
- `KSP_PCApplyBAorAB` returns `B⁻¹·A·in` — algorithm sees `M = B⁻¹·A`.
- Algorithm-internal "x" IS `x_user` (no variable change). The
  algorithm tracks the **preconditioned** residual
  `r_pre = B⁻¹·(b − A·x)`.
- **Required at solve start:** apply `B⁻¹` to `bLocal_unpre =
  b − A·x_initial` once, to get `bLocal_pre`. Algorithm starts with
  `r = bLocal_pre`. Without this, `r = b − A·x_initial`, which is
  inconsistent with the M-space algebra the cycles use.
- Snapshot trueres computation (`MatMult(A, x_total)`) is correct as-is
  for PC_LEFT — `x_total = x_user`, no unwrap.
- `iterres` reported in the trace is the **preconditioned** norm; this
  is the documented PETSc convention for `KSP_NORM_PRECONDITIONED`.

### PC_SYMMETRIC (deferred to Phase 4c)
- Setup: split preconditioner `B = B_L · B_R`, solve
  `B_L⁻¹·A·B_R⁻¹·y = B_L⁻¹·b` where `y = B_R·x_user`.
- Required at solve start: apply `B_L⁻¹` to bLocal.
- Required at solve end: apply `B_R⁻¹` to `x_alg` (or equivalent).
- Used mainly for ICC / Cholesky on symmetric operators. Test problems
  in Phase 5a are nonsymmetric (cdr_small has advection), so PC_SYMMETRIC
  is rare. Implement minimally; one tripwire on a symmetric variant.

## Sub-phases

### Phase 4a — PC_RIGHT correctness

**Code changes:**

1. Add `Vec x_initial_guess` to `KSP_GMSTAB`. Lazily allocated when
   `pc_side == PC_RIGHT` (or `PC_SYMMETRIC`); destroyed at top of every
   `KSPSolve_GMSTAB` and in `KSPReset_GMSTAB`. (Per Phase 3d audit
   pattern A: every per-solve allocation must reset.)
2. At `KSPSolve_GMSTAB` top, after the destroy-then-VecDuplicate block,
   conditionally save `x_initial_guess`:
   ```c
   PCSide pc_side; KSPGetPCSide(ksp, &pc_side);
   if (pc_side == PC_RIGHT || pc_side == PC_SYMMETRIC) {
     PetscCall(VecDuplicate(b, &gms->x_initial_guess));
     PetscCall(VecCopy(x_local, gms->x_initial_guess));
   }
   ```
3. Introduce `KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local)`
   that:
   - For `PC_NONE` / `PC_LEFT`: does `x_local += x_global` (current
     behavior).
   - For `PC_RIGHT`: computes `x_local = x_initial + B⁻¹·(x_global +
     x_local − x_initial)`.
4. Replace **every** in-line `VecAXPY(x_local, 1.0, gms->x_global)` at
   the 7 return paths with a call to `FinalizeSolution_Private`. (Per
   Phase 3d audit pattern B: enumerate every exit, audit the
   post-condition.)
5. Update `KSPGMSTABSnapshot_Private` so that for `PC_RIGHT` the
   `norm_true` recomputation unwraps `x_total → x_user` before
   `MatMult(A, ...)`.

**Tripwires (sequential + n=2,4,8):**

- `ex_gmstab_pcright_jacobi.c` — load cdr_small, set `PCJACOBI`,
  `PC_RIGHT`, `KSP_NORM_UNPRECONDITIONED`. Solve. Gates:
  - `KSPGetConvergedReason == KSP_CONVERGED_ATOL`
  - externally measured `‖b − A·x_returned‖ ≤ 1.5·tolabs`
  - internal rnorm matches external rnorm to FP precision (catches
    "user got `x_alg` instead of `x_user`" — this is the failure mode
    if the unwrap is missing)
- `ex_gmstab_pcright_nzg.c` — same but with `KSPSetInitialGuessNonzero`
  and `x0 = 0.7·𝟙`. Verifies the `x_initial` save path.

### Phase 4b — PC_LEFT correctness

**Code changes:**

1. At `KSPSolve_GMSTAB` top, after the unpreconditioned `bLocal = b −
   A·x_initial` computation:
   ```c
   if (pc_side == PC_LEFT) {
     Vec b_pre;
     PetscCall(VecDuplicate(b, &b_pre));
     PetscCall(KSP_PCApply(ksp, gms->b_local, b_pre));
     PetscCall(VecCopy(b_pre, gms->b_local));
     PetscCall(VecDestroy(&b_pre));
   }
   PetscCall(VecCopy(gms->b_local, gms->r));   /* r = b_pre */
   ```
   *(Note: this `KSP_PCApply` is the only PCApply Phase 4b adds to the
   solve loop. The cycles continue to use `KSP_PCApplyBAorAB` which
   already handles dispatch.)*
2. The dir_rbio rebuild matvec at the natural-flow restart/replace path
   uses `KSP_PCApplyBAorAB(ksp, x_local, Ax_drv, ws.work_n)`. For
   PC_LEFT this gives `Ax_drv = B⁻¹·A·x_local`. Then `r = bLocal − Ax_drv =
   B⁻¹·(b − A·x_initial) − B⁻¹·A·x_local = B⁻¹·(b − A·(x_initial +
   x_local))`. This is the correct preconditioned residual — no code
   change needed here, just verify by walking the algebra.
3. `Snapshot_Private`'s trueres path: unchanged, the `MatMult(A,
   x_total) → r_true = b − Ax` produces the unpreconditioned residual,
   which is what trueres should be.

**Tripwires (sequential + n=2,4,8):**

- `ex_gmstab_pcleft_jacobi.c` — `PCJACOBI`, `PC_LEFT`. Same gates as 4a.
- `ex_gmstab_pcleft_bjacobi.c` — `PCBJACOBI` (the standard parallel
  default).

### Phase 4c — PC_SYMMETRIC (defer)

Implement only after 4a/b are landed and 4d's CPU sweep is green. Out
of scope for the initial Phase 4 pass; one ICC tripwire on a symmetric
variant (e.g. a Laplacian) when we revisit.

### Phase 4d — CPU PC variety sweep

After 4a + 4b are validated, run the full PC catalog over both pc_sides:

| PC type | rationale |
|---|---|
| `PCJACOBI` | simplest scalar PC — covered by 4a/4b focused tripwires |
| `PCBJACOBI` | block-Jacobi, the standard parallel default |
| `PCSOR` | red-black SOR sweep |
| `PCILU` | incomplete LU factorisation |
| `PCASM` | additive Schwarz |
| `PCGAMG` | algebraic multigrid (PETSc-native) |
| `PCHYPRE` (BoomerAMG) | algebraic multigrid (Hypre) |

**Tripwire:** `ex_gmstab_pc_sweep.c` — iterates over the cross-product
of `pc_side ∈ {RIGHT, LEFT}` × `pc_type ∈ {above list}`, runs
`KSPSolve(cdr_small)`, asserts the 3 gates per combination. Reports
per-combination PASS/FAIL plus an overall result.

### Phase 4e — GPU-resident PCs (deferred, separate phase)

CUDA-specific PCs (`PCAIJCUSPARSE` factorisations, `PCAMGX`, GPU
BoomerAMG, etc.) have a fundamentally different lifecycle — they hold
device memory across solves, often with their own state machines. The
"init-once-never-reset" pattern (Phase 3d Pattern A) is much more
likely to fire on these. Doing 4e separately means we can apply the
lessons from 4a–4d's audit before adding the GPU axis.

**Why split:** doing all of 4 + GPU together would conflate "PC algebra
is wrong on PC_LEFT" (4b) with "AMGX solver context isn't being reset
between solves" (4e), which are very different bugs that would
otherwise both surface in a combined phase.

## Risks (informed by Phase 3d findings)

The Phase 3d bug taxonomy in `PHASE3_STATUS.md` lists five patterns —
all five could surface in Phase 4. Audit checklist applied to Phase 4:

| Pattern | Phase 4 watchpoint |
|---|---|
| **A: Init-once-never-reset** | `gms->x_initial_guess` — must destroy at top of every `KSPSolve` (Phase 3d Bug 2). The PCApply on `bLocal` at solve start is per-solve work — must repeat every solve, not lazy-init (Bug 9 analog). |
| **B: Multi-path invariant violation** | The 7 return paths from Phase 3d each need to call `FinalizeSolution_Private`. New `Finalize` helper must produce correct `x_user` regardless of which path called it. This is the most direct re-occurrence risk. |
| **C: Snapshot rhythm divergence** | Adding the unwrap inside `Snapshot_Private` could subtly change semantics on edge cases (e.g. snapshot fired before `gms->x_initial_guess` is allocated). Audit every snapshot-emission site. |
| **D: Off-by-statement bookkeeping** | The order of `VecCopy(x_local, x_initial_guess)` vs `VecSet(x_local, 0)` matters — must save the initial guess **before** the zero. |
| **E: Latent uninitialised state** | `Vec x_initial_guess` must be NULL-initialised in `KSPCreate_GMSTAB` (PetscNew zeroes the struct, so this happens automatically — but verify). |

## New tripwires expected to add

| Validator | Phase | Coverage |
|---|---|---|
| `ex_gmstab_pcright_jacobi` | 4a | basic PC_RIGHT correctness |
| `ex_gmstab_pcright_nzg` | 4a | PC_RIGHT with non-zero initial guess (exercises x_initial save/restore) |
| `ex_gmstab_pcleft_jacobi` | 4b | basic PC_LEFT correctness |
| `ex_gmstab_pcleft_bjacobi` | 4b | parallel-default PC_LEFT |
| `ex_gmstab_pc_sweep` | 4d | umbrella CPU PC sweep |

Plus negative tests for the most likely regressions — temporarily revert
the unwrap in `FinalizeSolution_Private`, rerun `ex_gmstab_pcright_*`,
confirm tests fail with the documented signature; restore.

Each new tripwire follows the existing pattern from
`ex_gmstab_natural_nzg` (3-gate harness on KSP reason +
externally-measured residual + internal/external rnorm match).

## Estimated scope

- 4a: ~80 LOC + 2 tripwires + audit — half a day
- 4b: ~30 LOC + 2 tripwires + audit — half a day
- 4d: 1 sweep tripwire + 5 per-PC tripwires — ~1 day
- Audit pass + commit + push — half a day

Total: ~2-3 days, gated by tripwire pass.

## Definition of done

- 40/40 → ~50/50 tripwires pass
- Phase 3d's 40 still pass (no regression on PC_NONE path)
- Bug taxonomy section in `PHASE3_STATUS.md` extended with any Phase 4
  findings, classified into the existing five patterns
- One commit on `saustinp/petsc:ksp-gmstab`, pushed
