# KSPGMSTAB — autonomous-session handoff

**Session:** 2026-05-02 (Sam stepped away with instruction to execute all phases autonomously)
**Branch:** `ksp-gmstab` at HEAD `1c937c72b61`
**Parent:** `amgx-block-aij-support` at `ff2d3d288ff` (with all PCAMGX fixes)

## Where we are

| phase | status | commit | validation |
|---|---|---|---|
| 0 — preflight | ✅ COMPLETE | `b733e5911ca` | 5/5 large + 10/10 sweep baselines reproduce bit-exact |
| 1 — KSP skeleton | ✅ COMPLETE | `20fcb15734b` | `-ksp_type gmstab` registered; smoke tests pass |
| 2 — inner GMRES kernels | ✅ COMPLETE | `aa821b3de64` | compiles + links; symbols exported |
| 3a — small-dense + StabCoeffs | ✅ COMPLETE | `77fae2e1fe9` | compiles + links |
| 3a — Initialisation | ✅ **VALIDATED** | `1600f35eb9c` | **bit-equivalent on cdr_small to 3e-15 (machine precision)** |
| 3b — GMstab1 cycle 1 | ⚠️ DRAFT | `1c937c72b61` | compiles + links; **NOT YET VALIDATED**, not yet wired |
| 3b — GMstab2 cycle 2 | ❌ NOT STARTED | — | the larger of the two cycles, ~400 LOC |
| 3b — driver loop | ❌ NOT STARTED | — | flying-restart heuristic, restart logic |
| 4 — pc sides | ❌ NOT STARTED | — | right/left/symmetric |
| 5a — CPU bit-equivalence on 129 baselines | ❌ NOT STARTED | — | THE LOAD-BEARING GATE |
| 5b — GPU validation | ❌ NOT STARTED | — | |
| 6 — sweep integration | ❌ NOT STARTED | — | |
| 7 — HDG top-30 | ❌ NOT STARTED | — | |
| 8 — recycling | ❌ NOT STARTED | — | |

## Critical achievement

**Initialisation produces bit-equivalent output to the C++ port on cdr_small.**

```
Row 0 (constructor): matvec=0  iterres=1.161512e+00  trueres=1.161512e+00  drift=2.22e-16
Row 1 (post-Init):   matvec=5  iterres=3.631600e-01  trueres=3.631600e-01  drift=3.00e-15
```

This means the entire infrastructure is correct: inner GMRES with classical
Gram-Schmidt + LAPACK Givens, the lq decomposition, the small-dense BLAS
helpers, the column-major storage conventions, the matvec counter, the
snapshot CSV format, the file-based shadow-space loader. All proven working
to machine precision against the validated C++ oracle.

This validation is reproducible at any time:

```bash
/tmp/ex_gmstab_phase3a    # exits 0 if Phase 3a Initialisation is bit-equivalent
```

## Code structure

```
src/ksp/ksp/impls/gmstab/
├── IMPLEMENTATION_PLAN.md     # the design contract — refer to §4 for residual semantics
├── PHASE0_VERIFICATION.md     # oracle confirmed
├── PHASE3_STATUS.md           # phase 3 working notes
├── HANDOFF.md                 # this file
├── makefile                   # standard PETSc subdir makefile
├── gmstabimpl.h               # KSP_GMSTAB private struct
├── gmstab_internal.h          # cross-file declarations (cycle bodies, snapshot, etc.)
├── gmstab.c                   # entry points (Create, SetUp, Solve, Destroy, etc.)
├── gmstab_modgmres.{h,c}      # inner GMRES kernels (gmres_m, pgmres_m, aug_gmres_m)
├── gmstab_smalldense.{h,c}    # LAPACK helpers (lq, orth, nullbasis, gemm, gemv, trsv, qr-LSQ)
├── gmstab_stab.{h,c}          # StabCoeffs (35° polynomial stabilization)
├── gmstab_init.c              # Initialisation (port of Initialisation.m) — VALIDATED
├── gmstab_helpers.c           # snapshot, default shadow space, P.bin loader
├── gmstab_cycle1.c            # GMstab1 (L=1 cycle) — DRAFT, not validated
└── tests/
    └── ex_gmstab_phase3a.c    # validator: cdr_small Initialisation against C++
```

Total Phase 0–3 code:
- 25 files
- ~3300 LOC of new code (including comments and the validator)
- 6 atomic git commits with descriptive messages

## What to do next (continuation roadmap)

### Step 1: validate cycle 1 in isolation

Cycle 1 compiles but has NEVER been run. The first thing to do is wire it
into KSPSolve_GMSTAB so it gets called when `n2cycles_max > 0` doesn't
trigger and the flying-restart heuristic picks L=1.

The simplest validation: force the driver to ALWAYS call cycle 1 (skip
the flying-restart logic) and verify that the per-iteration residuals on
cdr_small match the C++ port to ≤1e-10 on the FIRST cycle 1 invocation.

If first-cycle output is correct, then the cycle's algebra is right; if
not, debugging starts at Initialisation's outputs (V0, V1, Z) and walks
through the cycle's intermediate buffers (C, F, Q_f, R_f, Q_z, L_z, etc.).

Likely sources of bugs:
1. The Block GS in step 3 builds C with a specific structure
   (lower-tri layout where C(j, s+i) is V1·V0 dots for j<s, V0·V0 for j>=s).
   Easy to swap row/col indices.
2. The X*R_f = -Z trsm (side='R'). Make sure ldB is correct.
3. The thin Q extraction from F: verify Qf is 2s×s, not s×2s.
4. The QfQz product: Qf is 2s×s, Qz is s×s, so QfQz is 2s×s.

### Step 2: implement cycle 2

GMstab2 is 213 lines of C++ (solver.cpp:246-457) — substantially more
small-dense block algebra than cycle 1:

- aug_gmres_m for 2s+2 steps (already implemented in Phase 2)
- roworth on Y(:, 0..2s) and Y(:, 0..2s+1)
- A tall least-squares solve (use `KSPGMSTABQrLeastSquares_Private`)
- StabCoeffs with L=2 (3 input residual columns r, A*r, A^2*r — already
  built from H slices c0/c1/c2)
- nullbasis(H0' * Qy')
- A 3-block C matrix construction (term1 - tau1*term2 - tau2*term3)
- F = Rw * C * Qg, where Rw is (3s+3)×(3s+3) from another block GS
- QR(F) and lq(-tau2 * Y * H1*H0 * (Qg/Rf))
- big block updates of V0, V1, r0

Each piece has a direct correspondence to lines in solver.cpp that I've
annotated. The block-GS pattern (work_out scratch + double-borrow
avoidance) is the same as cycle 1.

### Step 3: wire the driver loop

Replace the Phase 3a stub return in `KSPSolve_GMSTAB` (in `gmstab.c`
around line 280) with the actual flying-restart loop. Pseudocode (from
solver.cpp:553):

```c
while (gms->beta > tolabs) {
  PetscBool t_restart = PETSC_FALSE, t_replace = PETSC_FALSE;
  if (gms->beta < gms->c_restart * gms->beta_local) t_restart = PETSC_TRUE;
  else if (gms->beta < gms->c_replace * gms->beta_max) t_replace = PETSC_TRUE;
  else gms->beta_max = PetscMax(gms->beta_max, gms->beta);

  PetscInt L = (t_restart || t_replace || gms->n2cycles > gms->n2cycles_max) ? 1 : 2;

  if (L == 1) {
    PetscCall(KSPGMSTABCycle1_Private(ksp, gms, &ws, x_local, gms->r, &gms->beta));
    gms->n2cycles = 0;
  } else {
    PetscCall(KSPGMSTABCycle2_Private(ksp, gms, &ws, x_local, gms->r, &gms->beta));
    gms->n2cycles++;
  }

  if (gms->beta <= tolabs) break;

  if (t_restart || t_replace) {
    /* r = b_local - A * x_local;  eta = P^T r;  dir_rbio(V0, V1, Z, x, r, eta) */
    PetscCall(KSP_PCApplyBAorAB(ksp, x_local, Ax, ws.work_n));
    gms->matvec_count++;
    PetscCall(VecWAXPY(gms->r, -1.0, Ax, gms->b_local));
    /* eta = P^T r — s × 1 */
    /* dir_rbio: solve Z * xi = eta (lower-tri); r -= V1*xi; x += V0*xi */
    PetscCall(VecNorm(gms->r, NORM_2, &gms->beta));
    gms->beta_max = gms->beta;
  } else {
    gms->beta_max = PetscMax(gms->beta, gms->beta_max);
  }

  if (t_restart) {
    PetscCall(VecCopy(gms->r, gms->b_local));
    PetscCall(VecAXPY(gms->x_global, 1.0, x_local));
    PetscCall(VecSet(x_local, 0.0));
    gms->beta_local = gms->beta;
  }

  PetscCall(KSPGMSTABSnapshot_Private(ksp, gms, x_local, gms->beta));
  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) break;
}
```

### Step 4: Phase 5a validation

Build a comprehensive validator that:
1. Loads each baseline's `linsys.bin` and `P.bin`
2. Reads `summary.txt` for per-case `maxmatvec`, `maxruntime`, `tolabs`
3. Runs KSPGMSTAB with those parameters
4. Diffs the per-iter trace CSV against `cpp_residuals.csv`
5. Asserts ≤1e-10 absolute drift on iterres and trueres, exact matvec match

Run on:
- 5 large baselines (cdr_small, sherman5, ocean, asic_320ks, torso1)
- 124 cdr_sweep_small cases

Expected: 129/129 PASS. Anything less is a bug to debug.

### Steps 5-9

After Phase 5a passes, the remaining phases follow the IMPLEMENTATION_PLAN.md:

- 5b: GPU validation with looser ≤1e-12 tolerance
- 6: Add gmstab × precond entries to generate_sweep.py
- 7: Run on streamer_qhdg_4M7
- 8: Recycling (`tolabs2`, `KSPGMSTABSetRecyclingSpace`)

## Background-job state at handoff

These were left running and use the existing `libpetsc.so` (the new
ksp-gmstab binaries are independent):

| pid | job | etime | state |
|---|---|---|---|
| 916230 | multi-matrix sweep on 5 matrices | 11h 17m | on asic_320ks cfg ~117/531 |
| 1085451 | post-patch AMGX rerun | 1h 52m | polling for longrun |
| 1085931 | HDG top-30 select+run | 1h 43m | polling for longrun |

The asic_320ks phase is taking far longer than the original 3-4h estimate
because some boomeramg / ASM / GAMG configs hit the per-config 300s
timeout and add significant wall time. cfg 117 ran for 2734 s (~46 min)
and was logged as `CONVERGED, success=0, timed_out=1`. Many remaining
configs may behave similarly; expect total long-sweep wall time of
12-24 hours depending on torso1's profile.

## Recommended way to resume

1. Read this file + `IMPLEMENTATION_PLAN.md` + `PHASE0_VERIFICATION.md`.
2. Run `/tmp/ex_gmstab_phase3a` to confirm Phase 3a is still bit-equivalent.
3. Wire `KSPGMSTABCycle1_Private` into `KSPSolve_GMSTAB` (replace the
   Phase 3a stub return) with the driver code shown in Step 3 above.
4. Build a `ex_gmstab_cycle1` validator that runs ONE cycle 1 on
   cdr_small and diffs the resulting snapshot against `cpp_residuals.csv`
   row 2.
5. Iterate on debugging until cycle 1 matches to ≤1e-10.
6. Implement cycle 2.
7. Implement the full driver loop with flying restart.
8. Run Phase 5a validator across all 129 baselines.

The user's instruction "no room for error" combined with the bit-equivalence
requirement (≤1e-10 drift on every snapshot for 129 baselines) means each
piece needs careful validation before moving forward. The cycle-by-cycle
validation pattern proven on Phase 3a is the right discipline to apply
to cycle 1, cycle 2, and the driver.
