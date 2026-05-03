# Phase 4 Follow-up — Action Plan (task #68)

**Status:** Plan saved; execution gated on Phase 5a (#50) completing first
**Date drafted:** 2026-05-03
**Predecessors:** Phase 4 (#49 done), Phase 4c (#69 done), **Phase 5a (#50) — must run first**
**Why deferred:** Phase 5a runs the full 129-baseline bit-equivalence sweep against the
current code (with the inline checks intact). This establishes the canonical "what does
correct bit-equivalence look like" diff before we change anything. Once that baseline
is captured, this patch can land and a re-run of 5a will confirm the only delta is the
expected snapshot-skipping rows that the diff-script filter handles.

This action plan supersedes `PHASE4_FOLLOWUP_PLAN.md` for the implementation phase
(that document remains as design rationale and cold-start context). The two are
complementary: read PHASE4_FOLLOWUP_PLAN.md for the *why* (the design discussion),
read this for the *what to do, in order*.

---

## 1. Scope (what this patch fixes, in one paragraph each)

### Fix A — drop inline `beta < ksp->abstol` short-circuits at 7 sites

The cycle's natively-tracked `beta` is in algorithm-native algebra (`||r_pre||` under
PC_LEFT or PC_SYMMETRIC; `||r_unprec||` under PC_RIGHT or PC_NONE). The user's
`ksp->abstol` is paired with whatever `KSPSetNormType` they chose. When the cycle
short-circuits on `beta < ksp->abstol` but the user asked for a different norm, the
exit fires in stale algebra. Most acute manifestation: `PC_LEFT + KSP_NORM_UNPRECONDITIONED + Jacobi`
on cdr_small reports `KSP_DIVERGED_ITS` with rnorm stalled just above tol. Fix: delete
the inline checks; rely on `ksp->reason` (set by `KSPGMSTABSnapshot_Private` →
`KSPConvergedDefault`, which honors normtype) to drive cycle exit. Two of the seven
sites are pre-Snapshot snapshot-skipping checks; removing them adds one trace row at
convergence under natural pairings. Mitigated by a 5-line filter rule in the diff
scripts.

### Fix B — `PC_RIGHT + KSP_NORM_PRECONDITIONED` reports wrong norm

Under `PC_RIGHT`, the algorithm natively tracks `beta = ||b - A·x||` (the unprec
residual). When the user requests `KSP_NORM_PRECONDITIONED`, they want
`||B⁻¹·(b - A·x)||`, but `gmstab_helpers.c:245` unconditionally returns `iter_norm`
(= `gms->beta` = `||r_unprec||`). The user gets the wrong scalar fed to
`KSPConvergedDefault` and reported via `KSPGetResidualNorm`. Wrong-answer bug —
silent, declared supported (`gmstab.c:1035`, priority 2). Fix: in
`KSPGMSTABSnapshot_Private`'s normtype dispatch, branch on PC side under
`KSP_NORM_PRECONDITIONED`: PC_LEFT/PC_SYMMETRIC keep using `iter_norm` (algorithm-native);
PC_RIGHT applies `PCApply(B⁻¹, r_true, r_pre_tmp)` and norms `r_pre_tmp`. PC_NONE is
also correct as-is since `||r_unprec|| = ||r_pre||` when B = I.

### Why bundled

Both fixes touch `KSPGMSTABSnapshot_Private`'s normtype/PC-side dispatch. Same
function, same code section, same review surface. Splitting them would mean two
patches that touch overlapping lines, double the review effort, and double the bit-eq
revalidation runs. Bundle.

---

## 2. Pre-flight checks

Before starting the patch, confirm these are all true:

```
[ ] Current branch is ksp-gmstab
[ ] Current commit is d19c05dc968 or later (Phase 4c + USER_MANUAL global-options note)
[ ] ./run_gmstab_tripwires.sh shows 68/68 passing (baseline)
[ ] git status is clean (no uncommitted gmstab/ changes)
[ ] Phase 4c is fully landed: PC_SYMMETRIC tripwire exists at tests/ex_gmstab_pcsymmetric_sweep.c
```

Run these to verify:
```bash
cd /home/sam/hpc_stack/petsc/src/ksp/ksp/impls/gmstab/tests && SKIP_BUILD=1 ./run_gmstab_tripwires.sh 2>&1 | tail -3
cd /home/sam/hpc_stack/petsc && git log --oneline | head -5
git -C /home/sam/hpc_stack/petsc status -s src/ksp/ksp/impls/gmstab/
```

Expected:
```
[tripwires] OVERALL: 68/68 passed
d19c05dc968 ... (most recent gmstab commit)
(empty — no uncommitted gmstab files)
```

---

## 3. Site-by-site implementation (in order)

> **Stop here if Phase 5a hasn't run yet.** Verify task #50 is `completed` and the
> 129-baseline diff has been captured & archived under the pre-patch code before
> proceeding. If 5a is still pending, this plan is a saved reference — don't execute.

### Step 0 — capture a "before" trace for the bit-eq smoke check

Phase 5a must have run first and captured the 129-baseline bit-equivalence diff
under the *current* code (with the inline checks intact). That diff is the
canonical reference. Step 0 here is the local smoke variant: capture the cdr_small
natural-flow trace under the **pre-patch** code so we can confirm this patch's
delta against a known starting point even before the full 5a re-run at the end.

```bash
mkdir -p /tmp/phase4_followup_traces

# Before-patch trace under PC_LEFT + NORM_PRECONDITIONED (natural pairing — should be
# bit-identical to itself across this patch, since the deleted checks would have
# fired on the same condition that ksp->reason now triggers)
/tmp/ex_gmstab_natural -ksp_gmstab_trace_csv /tmp/phase4_followup_traces/before_pcleft_normprec.csv

# Before-patch trace under PC_RIGHT + NORM_UNPRECONDITIONED (the canonical / fastest
# pairing — should also be bit-identical)
/tmp/ex_gmstab_pcright_jacobi -ksp_gmstab_trace_csv /tmp/phase4_followup_traces/before_pcright_normunprec.csv
```

After the patch is applied, rerun these and `diff` the CSVs. Under the natural
pairings, expect zero meaningful divergence. Any unexpected difference is a bug
introduced by this patch (or a pre-existing one we're now exposing).

### Step 1 — Algorithm code: drop inline check at gmstab.c:310 (post-init duplicate)

```c
// Current state (gmstab.c lines 310-316, after Phase 4c shifts):
  if (gms->beta <= ksp->abstol) {                                 // ← DELETE this 6-line block
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
    PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

**Justification it's safe to delete:** lines 293-295 (immediately above this block) are:
```c
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));   // line 293
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {               // line 295
  /* exit path with FinalizeSolution + workspace destroy */
```
The Snapshot at 293 already calls `KSPConvergedDefault` via the normtype-aware switch.
If the user's requested rnorm is below atol, `ksp->reason = KSP_CONVERGED_ATOL` is
already set, and the reason check at 295 catches it. The `gms->beta <= ksp->abstol`
clause is purely a *redundant* check on the algorithm-native scalar.

**Edit to make:** delete lines 310-316 entirely (the entire `if` block).

### Step 2 — Algorithm code: drop inline check at gmstab.c:331 (force_l1_only)

```c
// Current state (gmstab.c lines 329-340):
    PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));   // line 329
    if (gms->beta <= ksp->abstol) {                                            // line 331 ← DELETE the beta clause
      ksp->reason = KSP_CONVERGED_ATOL;
    } else if (!ksp->reason) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp,
        "KSPSolve_GMSTAB force_l1_only: ran one Cycle1, beta=%.6e > tolabs=%.6e\n",
        (double)gms->beta, (double)ksp->abstol));
    }
```

**Edit to make:** transform to (removing the dependent `else if` collapses to a
standalone `if`):
```c
    PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));
    if (!ksp->reason || ksp->reason == KSP_CONVERGED_ITERATING) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp,
        "KSPSolve_GMSTAB force_l1_only: ran one Cycle1, beta=%.6e > tolabs=%.6e\n",
        (double)gms->beta, (double)ksp->abstol));
    }
```

**Justification:** The Snapshot at line 329 already sets reason if convergence was
reached. The remaining else-if branch handles the "ran one cycle, didn't converge,
didn't error otherwise" case. Logic-equivalent under the natural pairings; correct
under cross-pairings (no premature exit on stale norm).

### Step 3 — Algorithm code: drop inline check at gmstab.c:355 (force_l2_only)

Symmetric to Step 2. Apply identical transformation to the force_l2_only block.

### Step 4 — Algorithm code: drop inline check at gmstab_cycle1.c:118 (pre-Snapshot)

```c
// Current state (gmstab_cycle1.c lines 112-127):
  /* C++ solver.cpp:171-174 returns from cycle1 here WITHOUT emitting a
     perf.read when beta has already crossed below tolabs (the polynomial
     step at chk03 happens to bring beta into convergence). The driver
     then breaks at "if (beta <= tolabs)". Mirror that exactly: skip the
     mid-cycle snapshot in this branch, so the trace stays bit-equivalent
     under any path that exits the cycle here. */
  if (beta < ksp->abstol) {                                                  // ← DELETE these 4 lines
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Mid-cycle snapshot per the C++ port (solver.cpp:175-177). */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

**Critical:** This site is *pre*-Snapshot. The inline check intentionally skips emitting
the snapshot when beta < tolabs at this point, mirroring C++ `solver.cpp:171-174`.
Removing it WILL emit one extra trace row per cycle that exits via this path under
the natural pairing.

**Edit to make:** delete the 4 lines `if (beta < ksp->abstol) { ... }`. Update the
comment block above to reflect new behavior:

```c
  /* C++ solver.cpp:171-174 returns from cycle1 here WITHOUT emitting a
     perf.read when beta has already crossed below tolabs. Our PETSc port
     emits the snapshot unconditionally and lets KSPConvergedDefault (called
     inside Snapshot via the user's KSPSetNormType choice) gate cycle exit
     via ksp->reason. This is the option-(i) deviation from the C++
     reference: cleaner cross-pairing semantics (no stale-norm short-circuit
     on PC_LEFT + KSP_NORM_UNPRECONDITIONED), at the cost of one extra
     trace row at convergence sites. The diff scripts in tests/ filter
     this row to preserve bit-equivalence with the C++ baseline trace
     under natural pairings. See PHASE4_FOLLOWUP_ACTION_PLAN.md and
     diff_gmstab_dumps.py for details. */
  /* Mid-cycle snapshot per the C++ port (solver.cpp:175-177). */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

### Step 5 — Algorithm code: drop inline check at gmstab_cycle1.c:420 (post-Snapshot, combined)

```c
// Current state (gmstab_cycle1.c line 420):
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));         // line 419
  if (beta < ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) || inner_terminated) {
//    ^^^^^^^^^^^^^^^^^^^^ ← DROP this clause                                   line 420
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

**Edit to make:** drop only the leading `beta < ksp->abstol ||` clause. Result:
```c
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if ((ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) || inner_terminated) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

**Justification:** post-Snapshot site, so reason is already correct. `inner_terminated`
is a different signal (pGMRESm internal completion) — keep it. Restores symmetry with
gmstab_cycle1.c:124 which already has the clean form.

### Step 6 — Algorithm code: drop inline check at gmstab_cycle2.c:158 (pre-Snapshot)

Symmetric to Step 4. Apply identical transformation. Update the comment to match.

### Step 7 — Algorithm code: drop inline check at gmstab_cycle2.c:476 (post-Snapshot)

```c
// Current state (gmstab_cycle2.c line 476):
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));         // line 475
  if (beta <= ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)) {
//    ^^^^^^^^^^^^^^^^^^^^^ ← DROP                                              line 476
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
```

**Edit:** drop the leading `beta <= ksp->abstol ||` clause. Result mirrors cycle1.c:124
and the now-cleaned cycle1.c:420.

### Step 8 — Verify all inline checks deleted; build

```bash
grep -n "beta <= ksp->abstol\|beta < ksp->abstol" /home/sam/hpc_stack/petsc/src/ksp/ksp/impls/gmstab/*.c
# Expected: empty output (zero hits)
cd /home/sam/hpc_stack/petsc && PETSC_DIR=$PWD PETSC_ARCH=arch-cuda-opt-i32 make libs 2>&1 | tail -3
# Expected: clean build, libpetsc.so.3.25.0 linked
```

If any inline check remains, locate it and apply the transformation. Don't skip this
verification — Pattern-B (multi-path invariant violation) is the failure mode if any
single site is missed.

### Step 9 — Run the existing tripwire suite (without diff filter yet)

Expected outcome at this stage: **most tripwires pass; the trace-equivalence diffs
(`diff_gmstab_dumps.py` for cycle1/cycle2, `diff_natural_seq_vs_par.py` for natural-flow)
will FAIL** because they'll see one extra trace row per snapshot-skipping site that
fired. This is the predicted divergence — confirms the snapshot-skipping deletions
took effect.

```bash
cd /home/sam/hpc_stack/petsc/src/ksp/ksp/impls/gmstab/tests && SKIP_BUILD=1 ./run_gmstab_tripwires.sh 2>&1 | tail -20
```

If the trace-diff tests pass at this stage *without* the filter, that means the
snapshot-skipping checks weren't being hit on the cdr_small problem in the first place
— in which case the algorithmic deletion is a pure no-op for natural pairings and we
don't need the filter. Note this and proceed to Step 12 (PC_RIGHT + NORM_PREC fix).

If the trace-diff tests fail with "+1 row at convergence", proceed to Step 10.

### Step 10 — Add the diff-script filter rule

Update three Python scripts:

#### `tests/diff_gmstab_dumps.py`

Find the row-by-row comparison loop. Before the loop, add:

```python
def filter_spurious_convergence_row(petsc_rows, cpp_rows, atol):
    """Phase 4 follow-up: drop a single trailing PETSc row whose iter_norm is below
    atol and that has no peer in the C++ trace. This compensates for the option-(i)
    deletion of pre-Snapshot inline `beta < ksp->abstol` checks at cycle1.c:118 and
    cycle2.c:158: those checks suppressed an end-of-cycle snapshot in the C++
    reference; we now emit the snapshot unconditionally and let KSPConvergedDefault
    gate exit. The trace gets one extra row per occurrence; this filter drops them.

    Conservative: only drops rows that BOTH (a) have iter_norm < atol AND
    (b) appear at positions where the PETSc trace has more rows than C++.
    Won't mask a real mid-trace divergence."""
    while len(petsc_rows) > len(cpp_rows):
        last = petsc_rows[-1]
        if last['iter_norm'] < atol:
            petsc_rows = petsc_rows[:-1]
        else:
            break
    return petsc_rows
```

Call it just before the row-by-row diff loop. The exact integration point depends on
how the script currently parses the CSVs — read the script first to understand its
shape.

#### `tests/diff_natural_seq_vs_par.py`

Same filter, same integration. This script compares two PETSc traces (sequential vs
parallel), so the spurious row would appear in BOTH and the filter is a no-op there.
Including it for consistency / future-proofing.

#### `tests/diff_seq_vs_parallel.py`

Same filter, same integration. Same no-op rationale as the natural variant.

### Step 11 — Re-run tripwires, expect 68/68

```bash
SKIP_BUILD=1 ./run_gmstab_tripwires.sh 2>&1 | tail -3
# Expected: [tripwires] OVERALL: 68/68 passed
```

If trace-diffs still fail, the filter logic isn't matching the actual trace shape —
debug by reading the failing diff output and adjusting the filter.

### Step 12 — Algorithm code: fix PC_RIGHT + NORM_PRECONDITIONED in Snapshot_Private

Current state (`gmstab_helpers.c` lines 240-262):

```c
  PetscReal rnorm_for_check;
  switch (ksp->normtype) {
  case KSP_NORM_UNPRECONDITIONED: rnorm_for_check = norm_true; break;
  case KSP_NORM_PRECONDITIONED:   rnorm_for_check = iter_norm; break;
  case KSP_NORM_NONE:             rnorm_for_check = 0.0;       break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
            "KSPGMSTAB: norm type %s not supported", KSPNormTypes[ksp->normtype]);
  }
  ksp->rnorm = rnorm_for_check;
```

**Issue:** under `(PC_RIGHT, KSP_NORM_PRECONDITIONED)`, `iter_norm = gms->beta = ||r_unprec||`,
not the prec norm. The user gets the wrong scalar.

**Constraint:** the existing code destroys `r_true` at line 223 (before line 240).
The fix needs `r_true` alive to apply `B⁻¹` to it. Two options:
- Move the `VecDestroy(&r_true)` to after the switch.
- Do the PCApply inside the `if (gms->snapshot_count > 0)` block where r_true exists.

**Choose option 2** — keeps r_true's lifetime tightly scoped, matches existing
structure. Compute the prec norm conditionally inside the same block where norm_true
is computed. Result:

```c
  PetscReal norm_true = 0.0;
  PetscReal norm_pre  = 0.0;   /* preconditioned residual norm under PC_RIGHT */
                                /* — only computed when needed; left at 0 otherwise */

  if (gms->snapshot_count == 0) {
    norm_true = iter_norm;
    /* Constructor snapshot: norm_pre defaults to iter_norm too — see Q&A below. */
    norm_pre = iter_norm;
  } else {
    Vec b, Ax, r_true;
    /* ... existing setup ... */
    PetscCall(MatMult(Amat, x_for_mult, Ax));
    PetscCall(VecWAXPY(r_true, -1.0, Ax, b));
    PetscCall(VecNorm(r_true, NORM_2, &norm_true));

    /* NEW (Phase 4 follow-up bug B): compute preconditioned residual norm
       under PC_RIGHT, where the algorithm natively tracks ||r_unprec|| but
       the user requested ||B⁻¹·r||. Mirrors cg.c:170-172 pattern. Only fires
       on the path that's actually needed — costs one extra PCApply per
       snapshot under PC_RIGHT + KSP_NORM_PRECONDITIONED, zero overhead
       under any other (side, normtype) combo. */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED && ksp->pc_side == PC_RIGHT) {
      Vec r_pre_tmp;
      PetscCall(VecDuplicate(b, &r_pre_tmp));
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCApply(pc, r_true, r_pre_tmp));
      PetscCall(VecNorm(r_pre_tmp, NORM_2, &norm_pre));
      PetscCall(VecDestroy(&r_pre_tmp));
    }

    if (x_unwrap) PetscCall(VecDestroy(&x_unwrap));
    PetscCall(VecDestroy(&Ax));
    PetscCall(VecDestroy(&r_true));
  }
```

Then update the switch:

```c
  PetscReal rnorm_for_check;
  switch (ksp->normtype) {
  case KSP_NORM_UNPRECONDITIONED:
    rnorm_for_check = norm_true;
    break;
  case KSP_NORM_PRECONDITIONED:
    /* Phase 4 follow-up: dispatch on PC side under NORM_PRECONDITIONED.
       PC_LEFT and PC_SYMMETRIC: iter_norm IS ||r_pre|| natively (algorithm
                                  operates in M-space and tracks the prec
                                  residual via bLocal preconditioning).
       PC_RIGHT:                  iter_norm is ||r_unprec||; must apply
                                  B⁻¹ to recover the prec norm. norm_pre
                                  was computed for this case in the
                                  snapshot_count > 0 block above; for the
                                  constructor snapshot we use iter_norm
                                  (= ||b|| = norm of unprec residual at x=0,
                                  which under PC_RIGHT equals beta_init,
                                  the algorithm's tracked initial residual).
       PC_NONE:                   ||r_pre|| = ||r_unprec|| = iter_norm. */
    if (ksp->pc_side == PC_RIGHT && gms->snapshot_count > 0) {
      rnorm_for_check = norm_pre;
    } else {
      rnorm_for_check = iter_norm;
    }
    break;
  case KSP_NORM_NONE:
    rnorm_for_check = 0.0;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
            "KSPGMSTAB: norm type %s not supported", KSPNormTypes[ksp->normtype]);
  }
  ksp->rnorm = rnorm_for_check;
```

**Subtle point — the `gms->snapshot_count > 0` guard:** the constructor snapshot is
called with `iter_norm = ||b||`, and `norm_pre` is uninitialized in that call path.
Under `(PC_RIGHT, KSP_NORM_PRECONDITIONED)` at the constructor snapshot, falling back
to `iter_norm = ||b||` is reasonable — at x=0, `||r_pre|| = ||B⁻¹·b||`, but applying
B⁻¹ to b just for the constructor row is overkill (PETSc's GMRES doesn't bother
either). The constructor snapshot is a "starting state" record; its rnorm is mostly
informational. Tripwires test convergence at later iterations.

If we later decide we want bit-equivalent constructor norms under
`(PC_RIGHT, KSP_NORM_PRECONDITIONED)`, add an else branch in the constructor block
that applies B⁻¹ to b. For now, leave it.

### Step 13 — Add tripwire for PC_RIGHT + NORM_PRECONDITIONED

Create `tests/ex_gmstab_pcright_normpre.c`. Structure modeled on
`ex_gmstab_pcright_jacobi.c` but with `KSPSetNormType(KSP_NORM_PRECONDITIONED)` and
an additional gate verifying the reported `rnorm_internal` matches an externally
computed `||B⁻¹·(b - A·x)||` to FP precision.

External-check implementation:
```c
Vec Ax, r_true, r_pre;
PetscCall(VecDuplicate(b, &Ax));
PetscCall(VecDuplicate(b, &r_true));
PetscCall(VecDuplicate(b, &r_pre));
PetscCall(MatMult(A, x, Ax));
PetscCall(VecWAXPY(r_true, -1.0, Ax, b));
PetscReal user_unprec_res;
PetscCall(VecNorm(r_true, NORM_2, &user_unprec_res));
/* Apply B⁻¹ externally to verify the algorithm's internal rnorm. */
PC pc;
PetscCall(KSPGetPC(ksp, &pc));
PetscCall(PCApply(pc, r_true, r_pre));
PetscReal user_prec_res;
PetscCall(VecNorm(r_pre, NORM_2, &user_prec_res));
```

Gates:
```c
int g_reason = (reason == KSP_CONVERGED_ATOL);
int g_user_unprec = (user_unprec_res <= TOL_UNPREC);  /* sanity — solution is correct */
/* CRITICAL: the algorithm's reported rnorm under (PC_RIGHT, NORM_PREC) must equal
   the externally-computed ||B⁻¹·r||. This is the gate that catches the Bug B
   regression (current code reports ||r_unprec|| instead of ||r_pre||). */
int g_self_prec = (fabs(rnorm_internal - user_prec_res)
                     <= 1e-9 + 0.1 * fabs(user_prec_res));
```

If Bug B's pre-fix code is in place, `g_self_prec` would FAIL because
`rnorm_internal = ||r_unprec||` while `user_prec_res = ||B⁻¹·r_unprec||`. After the
fix, these match.

Add the test to `tests/run_gmstab_tripwires.sh` VALIDATORS and PARALLEL_VALIDATORS
arrays, mirroring how `ex_gmstab_pcright_jacobi` was added.

### Step 14 — Optionally revert GATE_TOL workarounds in PC_LEFT tests

The four PC_LEFT-related tripwires currently use `GATE_TOL = 1e-8` workaround
(introduced in Phase 4 audit) to absorb the cross-pairing stall under
`PC_LEFT + KSP_NORM_UNPRECONDITIONED + weak PC`. With this patch's Fix A, that stall
is gone — the algorithm now lets KSPConvergedDefault decide based on the user's atol.

Files using GATE_TOL:
- `tests/ex_gmstab_pcleft_jacobi.c`
- `tests/ex_gmstab_pcleft_bjacobi.c`
- `tests/ex_gmstab_pc_sweep.c`
- `tests/ex_gmstab_pc_multisolve.c`

For each, restore `TOLABS = 1e-10` and remove `GATE_TOL`. Convergence for
`PC_LEFT + Jacobi` may take many iterations (we measured ~48k under the algorithm's
behavior) — keep `max_it` high (e.g., 60000) for that single test.

If convergence to 1e-10 is too slow for the CI budget, keep the GATE_TOL workaround
on `pcleft_jacobi` *only* (it's a known-pathological case) and revert on the other
three.

This is a "nice to have" not a "must have" — the algorithm correctness is unchanged
either way; this is just cleaning up test-side workarounds. Ship the patch even if
this step gets deferred.

### Step 15 — Run full suite

```bash
cd /home/sam/hpc_stack/petsc/src/ksp/ksp/impls/gmstab/tests && SKIP_BUILD=1 ./run_gmstab_tripwires.sh 2>&1 | tail -10
```

Expected: **69/69 passing** (was 68/68; +1 from `ex_gmstab_pcright_normpre`).

If less than 69 pass, debug per Section 5.

### Step 16 — Smoke-test PC_LEFT + Jacobi on cdr_small (the original stall case)

```bash
/tmp/ex_gmstab_pcleft_jacobi -ksp_atol 1e-10 -ksp_max_it 100000 2>&1 | tail -10
```

Expected: `reason=3` (KSP_CONVERGED_ATOL), `rnorm_internal < 1e-10`. Pre-patch this
returned `reason=-3` with rnorm stalled at ~9.7e-10. Confirms Fix A is producing the
intended behavior change.

### Step 17 — Capture "after" trace and diff against "before"

```bash
/tmp/ex_gmstab_natural -ksp_gmstab_trace_csv /tmp/phase4_followup_traces/after_pcleft_normprec.csv
/tmp/ex_gmstab_pcright_jacobi -ksp_gmstab_trace_csv /tmp/phase4_followup_traces/after_pcright_normunprec.csv

diff /tmp/phase4_followup_traces/before_pcleft_normprec.csv /tmp/phase4_followup_traces/after_pcleft_normprec.csv | head -30
diff /tmp/phase4_followup_traces/before_pcright_normunprec.csv /tmp/phase4_followup_traces/after_pcright_normunprec.csv | head -30
```

Expected:
- Under PC_LEFT + NORM_PREC: at most one extra row per cycle that previously hit a
  snapshot-skipping site. If cdr_small never hit those sites, no diff at all.
- Under PC_RIGHT + NORM_UNPREC: identical traces (this configuration is unaffected
  by Fix A and Fix B — algorithm-native is exactly what user asked for).

If unexpected divergence appears, do not commit — debug first.

### Step 18 — Update docs

#### USER_MANUAL.md

- §3 ("Warning — PC_LEFT + KSP_NORM_UNPRECONDITIONED cost and stall"): the stall part
  is now resolved. The 4–6 MatMults/cycle cost remains. Update the section title and
  body to reflect: cost still exists; stall is fixed; convergence may take more iters
  under weak PC, but reaches atol cleanly.
- §8 (Known limitations): remove or update the corresponding bullet about
  "PC_LEFT + KSP_NORM_UNPRECONDITIONED + weak PC stall".
- New §3 sub-paragraph noting the PC_RIGHT + KSP_NORM_PRECONDITIONED fix and the
  one-PCApply-per-snapshot cost under that pairing.

#### PHASE4_FOLLOWUP_PLAN.md

- Mark fixes A and B as implemented.
- Update the "site-by-site reference" appendix with completed status.

#### PHASE3_STATUS.md

- Add a new section noting the inline short-circuit cleanup and the PC_RIGHT prec
  norm fix. Cross-reference to this action plan.

### Step 19 — Commit and push

Two commits, atomic:

**Commit 1** — algorithm changes (Steps 1–7 + Step 12):
```
gmstab: phase 4 follow-up — drop inline beta<abstol short-circuits + fix PC_RIGHT NORM_PREC

[Body explains both fixes; references this plan]
```

**Commit 2** — test infrastructure (Steps 10, 13, 14, 18):
```
gmstab: phase 4 follow-up — diff-script filter + PC_RIGHT NORM_PREC tripwire + doc updates
```

Splitting is optional; one combined commit is also fine. The atomicity matters for
git bisect if a regression is later discovered.

### Step 20 — Update tasks

```
TaskUpdate #68 status=completed
TaskUpdate #50 (Phase 5a): no longer blocked by anything Phase-4-side; ready to start
```

---

## 4. Validation gates (must all pass before declaring done)

| Gate | Expected | Verify with |
|---|---|---|
| All 7 inline checks deleted | grep returns 0 hits | `grep -n "beta <= ksp->abstol\|beta < ksp->abstol" gmstab*.c` |
| libpetsc builds clean | no errors | `make libs` |
| 69/69 tripwires pass | 17 sequential + 48 parallel + 4 trace-diff = 69 (was 68; +1 from new pcright_normpre) | `./run_gmstab_tripwires.sh` |
| ex_gmstab_pcright_normpre passes at all rank counts | g_reason && g_user && g_self_prec | embedded in suite |
| PC_LEFT + Jacobi reaches ATOL convergence | reason=3, rnorm <= 1e-10 | manual smoke test from Step 16 |
| PC_RIGHT + UNPREC trace bit-equivalent before/after | empty diff | Step 17 |
| PC_LEFT + PREC trace bit-equivalent before/after (with filter) | filter consumes any +N rows | Step 17 |

---

## 5. Pattern-based audit checklist (re-verify after implementation)

| Pattern | Risk for this patch | Verification |
|---|---|---|
| **A** — Init-once-never-reset | Fix B introduces `norm_pre` local; allocated/destroyed per snapshot. Verify VecDestroy on every path that VecDuplicates `r_pre_tmp`. | Read the modified Snapshot_Private end-to-end |
| **B** — Multi-path invariant violation | All 7 inline checks deleted at once; if any are missed, behavior diverges between affected and unaffected sites under cross-pairings | `grep` verification at Step 8 |
| **C** — Lazy-init dispatch cascade | Cycle bodies still use KSP_PCApplyBAorAB; this patch doesn't introduce new dispatch paths | Read cycle bodies — no changes there |
| **D** — Snapshot rhythm divergence | Fix B changes Snapshot_Private's reported `ksp->rnorm` under PC_RIGHT + NORM_PREC. Verify that `KSPConvergedDefault` then sees the corrected scalar and `g_self_prec` in the new tripwire confirms internal/external agree | Tripwire `g_self_prec` gate |
| **E** — Off-by-statement bookkeeping | The `norm_pre` computation must happen INSIDE the `if (snapshot_count > 0)` block where r_true exists. If hoisted out, r_true is uninitialized | Code review at Step 12 |
| **F** — Latent uninitialised state | `norm_pre = 0.0` initialization at top of function avoids reading uninitialized scalar in the constructor-snapshot fall-through path | Initialize at declaration; verify by reading |

---

## 6. Risks and mitigations (updated for current sequencing)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phase 5a baseline diff hasn't been established before this patch — can't isolate this patch's impact from pre-existing baseline drift | None (resolved by sequencing) | Patch is gated on Phase 5a (#50) completing first. Step 0 still useful as a local smoke check during execution, but the canonical diff comes from 5a |
| Filter logic in diff scripts incorrectly drops a real divergence | Low | Filter explicitly checks (a) `iter_norm < atol` AND (b) PETSc has more rows than C++ AND (c) row is at end of trace. Three conjunctive conditions — hard to misfire |
| `pcleft_jacobi` test becomes too slow at `TOLABS=1e-10` (Step 14) | Medium | Keep GATE_TOL workaround on that one test only; document |
| Pre-existing PCBJACOBI(default sub-PC ILU) failures under PC_SYMMETRIC are unrelated to this patch but will be visible | High | Already documented in USER_MANUAL.md §8; not a regression — pre-existing PETSc-side bug |
| `norm_pre` PCApply errors under PCs that don't implement applysymmetricright (irrelevant — we use PCApply not PCApplySymmetricRight here) | None | Sanity-checked: line 12's PCApply(pc, r_true, r_pre_tmp) uses regular PCApply which all PCs implement |
| Future port additions re-introduce `beta < ksp->abstol` pattern | Low (one-time risk) | Add a code-review checklist note: "no new inline `beta < ksp->abstol` short-circuits in cycle bodies; route exit through ksp->reason" |

---

## 7. Cold-start checklist (resumable from no context)

If picking this up after a context loss:

1. Read `PHASE4_FOLLOWUP_PLAN.md` §1–§3 for the design rationale (why we're doing this).
2. Read `PC_LEFT_NORM_DISCUSSION.md` Q1–Q20 for the full design discussion that led
   to this fix.
3. Read this file (`PHASE4_FOLLOWUP_ACTION_PLAN.md`) §1–§3 for the implementation
   blueprint.
4. Run pre-flight checks (§2 above) and confirm baseline.
5. Verify line numbers haven't drifted: re-run the grep at §3 Step 8.
6. Apply changes Step 1 → Step 17 in order. Do NOT skip the pre-patch trace capture
   at Step 0.
7. After Step 17, refer to validation gates at §4 to declare done.

---

## 8. What's intentionally out of scope

Listed in original `PHASE4_FOLLOWUP_PLAN.md` §8. Not bundling here:
- PC_SYMMETRIC convention reconciliation between gmstab and CG/GMRES (gmstab reports
  `||B_L⁻¹·r||` under PC_SYMMETRIC + NORM_PREC; CG would report `||(B_L·B_R)⁻¹·r||`).
  Defer to Phase 6 if anyone notices and complains.
- Reducing the 4–6 snapshots/cycle cost under cross-pairings (would be an algorithmic
  deviation from the C++ reference; breaks bit-equivalence intentionally).
- KSP_NORM_NATURAL support (gmstab doesn't declare it; default-error handling works).
- Reducing constructor-snapshot overhead under PC_RIGHT + NORM_PREC (the
  `gms->snapshot_count == 0` branch in Fix B uses iter_norm = ||b|| which is
  technically not the prec norm at x=0, but it's a starting-state record that
  doesn't affect convergence decisions).

---

## 9. Estimated effort

- Algorithm changes (Steps 1–7, 12): ~30 min
- Build + initial test run (Step 8–9): ~5 min
- Filter logic (Step 10): ~20 min (need to read the diff scripts first)
- New tripwire (Step 13): ~30 min
- Optional GATE_TOL revert (Step 14): ~15 min
- Validation runs (Steps 15–17): ~15 min
- Doc updates (Step 18): ~30 min
- Commit/push (Step 19): ~5 min

Total: ~2.5 hours from cold start. Probably half that if running through it
end-to-end without re-reading the design rationale.

---

## 10. Definition of done

All of the following are true:

- ✅ All 7 inline `beta < ksp->abstol` checks deleted; grep verifies 0 hits
- ✅ `Snapshot_Private` correctly dispatches under (PC_RIGHT, NORM_PRECONDITIONED) via
  one extra PCApply
- ✅ Diff scripts filter the spurious convergence-time row under natural pairings
- ✅ New tripwire `ex_gmstab_pcright_normpre` exists and passes at n=1/2/4/8
- ✅ Tripwire suite reports 69/69 passing
- ✅ `PC_LEFT + Jacobi + atol=1e-10` smoke test reaches ATOL (was DIVERGED_ITS pre-fix)
- ✅ Pre-patch vs. post-patch traces diff cleanly (only spurious-convergence rows)
- ✅ USER_MANUAL.md, PHASE4_FOLLOWUP_PLAN.md, PHASE3_STATUS.md updated
- ✅ Commits pushed to `saustinp/petsc:ksp-gmstab`
- ✅ Task #68 marked completed; Task #50 (Phase 5a) noted as ready to start
