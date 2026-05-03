# Phase 4 Follow-up — Drop Inline `beta < abstol` Short-circuits

**Status:** Planned, deferred until after Phase 5a baseline sweep completes.
**Tracked task:** #68 (blocked by #50).
**Author:** Sam Austin / Claude (paired discussion)
**Date drafted:** 2026-05-03

This plan resumes work that was scoped during the Phase 4 audit pass but deferred. It
is intentionally written so a future implementer (or a future Claude session with
no conversation history) can pick it up cold.

If reading this with no prior context: start with `PC_LEFT_NORM_DISCUSSION.md` for the
design rationale. This document is the *implementation* plan; the rationale is in the
discussion transcript.

---

## 1. TL;DR (≤ 100 words)

GMSTAB inherits a `if (gms->beta <= ksp->abstol) break` short-circuit pattern from its
MATLAB/C++ reference. PETSc exposes a `KSPSetNormType` axis the reference doesn't
have. When a user sets `PC_LEFT + KSP_NORM_UNPRECONDITIONED + weak PC`, the short-circuit
fires on the wrong norm and the algorithm reports `KSP_DIVERGED_ITS` with a residual
just above `atol` — even though `x` is numerically correct.

**Plan:** delete six inline `beta <= abstol` clauses. Rely on `ksp->reason`
(set by `KSPGMSTABSnapshot_Private` → `KSPConvergedDefault`, which honors normtype).
Add a one-rule filter in the trace-diff scripts to skip a single trailing convergence-time
row that the cleaner code emits but the C++ baseline doesn't.

---

## 2. Why we're doing this

### The bug class

Under `PC_LEFT`, the GMstab cycle internally tracks `||r_pre|| = ||B⁻¹·(b - A·x)||`.
The cycle's six inline short-circuits at `gmstab.c:287/308/332`, `cycle1.c:118`,
`cycle2.c:158/476` test `gms->beta = ||r_pre||` against `ksp->abstol`. This is correct
*in the MATLAB/C++ reference* because that reference has no `KSPSetNormType` API —
`tolabs` is implicitly an expectation about `||r_pre||` (the algorithm's natively
tracked scalar).

When PETSc exposes `KSPSetNormType` as an independent axis from `KSPSetPCSide`, that
implicit assumption breaks. A user can set:
- `KSPSetPCSide(ksp, PC_LEFT)` — algorithm tracks `||r_pre||`
- `KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED)` — convergence check uses `||r_unprec||`

Same `ksp->abstol` scalar, two different quantities being compared to it.

For Jacobi (`B = diag(A)`), the conditioning ratio between `||r_pre||` and
`||r_unprec||` runs ~10× on advection-dominated problems. The cycle short-circuits
when `||r_pre|| ≤ 1e-10` but `||r_unprec||` stalls at ~1e-9. KSPConvergedDefault sees
`rnorm > atol` and returns `ITERATING`. Outer loop launches another cycle, which
short-circuits again immediately. Loop spins until `max_it`.

Concrete reproduction (tripwire prior to GATE_TOL workaround):
```
[pcleft-jacobi] reason=-3 rnorm_internal=9.707882e-10 (stalled, exited via DIVERGED_ITS)
```

### What other PETSc solvers do

(Audit done; full results in `PC_LEFT_NORM_DISCUSSION.md` Q10.)

| Solver | Posture |
|---|---|
| GMRES, BiCGStab, iBiCGS, FGMRES, LGMRES | Refuse `PC_LEFT + NORM_UNPRECONDITIONED` at registration |
| CG, BiCG, Pipelined CG | Compute both norms on demand; no inline `beta < abstol` short-circuit |
| **gmstab (us)** | Allows the combo, but inline check uses stale algebra → stall |

The CG/BiCG family delegates exit decisions to `(*ksp->converged)()`, which respects
`ksp->normtype`. GMRES bars the combo entirely. We're the odd-one-out, due to porting
faithfulness to MATLAB/C++ which doesn't have this axis.

### What we're choosing

- **Not bar the combo.** GMRES's "refuse at registration" approach is viable but
  surprises users who reasonably expect `PC_LEFT + NORM_UNPRECONDITIONED` to work
  (CG accepts it). Phase 4 already declared `KSP_NORM_UNPRECONDITIONED + PC_LEFT`
  supported at `gmstab.c:999`; revoking that is a user-facing breaking change.
- **Match the CG family.** Delete the inline checks; route exit decisions through
  `ksp->reason`. This is the cleanest fix and keeps user choice intact.

### What this is *not*

- Not an algorithmic change. The IDR(s) recurrences, BGS, polynomial step, etc. are
  untouched. Only the cycle's *exit gating* changes.
- Not a performance improvement under the natural pairings. `PC_LEFT + NORM_PREC`
  and `PC_RIGHT + NORM_UNPREC` already work optimally; this fix doesn't change them.
- Not a fix for the conditioning-ratio fundamental — the algorithm still cannot drive
  `||r_unprec||` arbitrarily low under weak PC + PC_LEFT. It just lets the algorithm
  *try* until `KSPConvergedDefault` says stop, instead of bailing on `||r_pre||`.

---

## 3. Code changes

### 3.1 Algorithm code

Six sites total. Categorized in two groups:

#### Group A — post-Snapshot sites (4 sites): pure cleanup

These sites fire *after* a `Snapshot_Private` (or `SnapshotLocal_Private`) call has
already set `ksp->reason` via `KSPConvergedDefault`. The `beta <= ksp->abstol` clause
is redundant.

##### `gmstab.c:287` (after post-init duplicate snapshot at line 270)

```c
// REMOVE these 5 lines:
if (gms->beta <= ksp->abstol) {
  ksp->reason = KSP_CONVERGED_ATOL;
  PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
  PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Behavior preserved by:
// - line 270 already called Snapshot_Private (which sets reason via KSPConvergedDefault)
// - line 272 already has: if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) ...
//   which catches the same convergence under the natural pairing.
```

##### `gmstab.c:308` (force_l1_only validation path)

```c
// CURRENT:
if (gms->beta <= ksp->abstol) {
  ksp->reason = KSP_CONVERGED_ATOL;
} else if (!ksp->reason) {
  ksp->reason = KSP_DIVERGED_BREAKDOWN;
  ...
}

// CHANGE TO:
if (!ksp->reason || ksp->reason == KSP_CONVERGED_ITERATING) {
  ksp->reason = KSP_DIVERGED_BREAKDOWN;
  ...
}
```

(The `beta <= abstol` branch becomes `ksp->reason == KSP_CONVERGED_ATOL` set by the
preceding Snapshot at line 306. The else-if becomes the only explicit branch.)

##### `gmstab.c:332` (force_l2_only validation path) — symmetric to 308

Apply the same transformation as 308.

##### `gmstab_cycle2.c:476` — restore symmetry with `cycle1.c:124`

```c
// CURRENT (cycle2.c:475-481):
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
if (beta <= ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)) {
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// CHANGE TO (matching cycle1.c:123-127):
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

This is a port-hygiene cleanup — `cycle1.c:124` already has the right shape; cycle2
was missed during the Phase 3c reason-driven-loop refactor. Restoring symmetry was
called out in the discussion (Q18).

#### Group B — pre-Snapshot snapshot-skipping sites (2 sites): normtype-guard

These two sites are *before* a snapshot and the inline check intentionally skips
emitting it (mirroring C++ `solver.cpp:171-174` per comments at `cycle1.c:112-117`).
**Removing them naively would break bit-equivalence with the C++ baseline by emitting
one extra trace row.**

After discussion (Q19), the chosen approach is to **drop them entirely** and add a
filter in the trace-diff script (Group C below). Rationale: smaller blast radius in
the algorithm code.

##### `gmstab_cycle1.c:118-121` — drop entirely

```c
// REMOVE these 4 lines:
if (beta < ksp->abstol) {
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// KEEP the Snapshot + reason-check that already follows (lines 122-127):
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Update comment at lines 112-117 to reflect new behavior:
// /* C++ solver.cpp:171-174 returns from cycle1 here WITHOUT emitting a perf.read
//    when beta has already crossed below tolabs; our PETSc port emits the snapshot
//    unconditionally and lets KSPConvergedDefault (called inside Snapshot via the
//    user's KSPSetNormType choice) gate the exit. The diff scripts in tests/ skip
//    a single trailing convergence-time row to preserve bit-equivalence with the
//    C++ baseline trace under natural (PC_LEFT, NORM_PRECONDITIONED) pairing. */
```

##### `gmstab_cycle2.c:158-161` — drop entirely (symmetric to cycle1.c:118)

Apply the same transformation as cycle1.c:118.

#### Algorithm change summary

| Site | Group | Action |
|---|---|---|
| `gmstab.c:287` | A | Drop the entire `if` block (5 lines) |
| `gmstab.c:308` | A | Drop the `beta <= abstol` branch, simplify else-if to standalone if |
| `gmstab.c:332` | A | Same as 308 |
| `gmstab_cycle1.c:118` | B | Drop `if (beta < abstol) return` (4 lines); update comment |
| `gmstab_cycle1.c:420` | A | Drop `beta < ksp->abstol \|\|` clause; restore cycle1.c:124 symmetry |
| `gmstab_cycle2.c:158` | B | Same as cycle1.c:118 |
| `gmstab_cycle2.c:476` | A | Drop `beta <= abstol \|\|` clause; restore cycle1.c:124 symmetry |

**(Updated 2026-05-03 — final audit pass uncovered the missed 7th site at
`cycle1.c:420`. Original plan listed 6; correct count is 7.)**

Net diff: roughly 17 lines deleted, 6 lines of comment updated. No new conditionals,
no new branches, no new `ksp->normtype` reads.

### 3.2 Test tooling

#### `tests/diff_gmstab_dumps.py` — add convergence-time-row filter

The two snapshot-skipping sites (cycle1.c:118 and cycle2.c:158) being dropped means
the PETSc trace will emit one extra row at convergence under the natural pairing —
that row would have been absent in the C++ trace (because C++ returns *before* its
`perf.read`). Add a pre-diff filter to drop the spurious row:

```python
def filter_spurious_convergence_row(petsc_rows, cpp_rows, atol):
    """Drop a single trailing PETSc row whose iter_norm is below atol and that has
    no peer in the C++ trace. This is the option-(i) trace asymmetry: the C++
    reference suppresses snapshots when beta drops below tolabs at the cycle1.c:118
    / cycle2.c:158 sites; PETSc emits them and lets KSPConvergedDefault gate exit.
    Filter at the diff-tool level so the algorithm code stays simple."""
    if len(petsc_rows) == len(cpp_rows) + 1:
        last = petsc_rows[-1]
        if last['iter_norm'] < atol:
            return petsc_rows[:-1]
    return petsc_rows
```

Apply this filter inside the existing diff loop, before row-by-row comparison.

#### `tests/diff_natural_seq_vs_par.py` and `tests/diff_seq_vs_parallel.py`

Same filter, inserted at the same point. These scripts compare PETSc-vs-PETSc
(seq vs. parallel), but the spurious row would appear in both — so the filter is a
no-op there. Including it for consistency / future-proofing.

#### Tripwires (optional cleanup)

After Group A + Group B + filter are in, the four tripwires that currently use
`GATE_TOL = 1e-8` workaround can be reverted to `TOLABS = 1e-10`:

- `tests/ex_gmstab_pcleft_jacobi.c`
- `tests/ex_gmstab_pcleft_bjacobi.c`
- `tests/ex_gmstab_pc_sweep.c`
- `tests/ex_gmstab_pc_multisolve.c`

The split TOLABS / GATE_TOL constant in those files becomes redundant — the algorithm
will now drive `||r_unprec||` to `atol = 1e-10` cleanly (slowly under PC_LEFT + Jacobi,
but it will get there or fail honestly). Restore the original strict-1e-10 gate.

If the algorithm still can't reach 1e-10 within reasonable `max_it` for PC_LEFT +
Jacobi (because the conditioning-ratio fundamental still applies — option (i) doesn't
fix that), keep the GATE_TOL workaround for that *one* test only and document the
rationale clearly. Other tests (bjacobi, ilu, asm) should pass at 1e-10 cleanly.

---

## 4. Sequencing

This plan is **blocked by Phase 5a (task #50).** Reasoning (from discussion Q20):

1. Phase 5a runs the full 129 C++ baseline sweep against current code (with the
   inline checks intact). This establishes the canonical bit-equivalence diff.
2. Apply Phase 4 follow-up: code changes from §3.1, filter from §3.2.
3. Re-run Phase 5a. The diff should now show zero meaningful divergences once the
   filter handles the spurious-convergence-row case.

If Phase 5a shows unexpected divergences *before* this follow-up, debug those first.
Don't apply the follow-up until baseline state is well-understood. Otherwise we'd be
debugging two changes simultaneously.

**Optional pre-requisite:** also complete Phase 4c (PC_SYMMETRIC) and Phase 4e (GPU
PCs) before this follow-up if the team wants Phase 4 fully done before going back
to retroactive cleanups. Either order works — the follow-up doesn't touch
PC_SYMMETRIC or GPU paths.

---

## 5. Validation gates

After implementation, the follow-up is "done" when **all** of the following pass:

### 5.1 Tripwire suite — 64/64 still passing

Run `./tests/run_gmstab_tripwires.sh`. Expected: same 64/64 pass count.

Specific tripwires that should *materially change* under the follow-up:

- `ex_gmstab_pcleft_jacobi` (single-rank): rnorm_internal now drives below 1e-10
  (previously stalled at ~9.7e-10 with `DIVERGED_ITS`). Reason should now be
  `KSP_CONVERGED_ATOL`. Iter count may increase substantially (algorithm keeps
  refining instead of bailing) — likely O(10⁴–10⁵) iters.
- `ex_gmstab_pc_sweep` PC_LEFT × jacobi row: same trajectory change.
- `ex_gmstab_pcleft_bjacobi` (n=1,2,4): should converge at similar iter count to
  before (no change; bjacobi was already in the "natural pairing approximately
  works" regime).

If `pcleft_jacobi` iter count is too large (>50k) for CI budget, **keep `max_it`
high in that test only and accept the longer run** — it's the one configuration
where the algorithm needs to grind.

### 5.2 Phase 5a baseline sweep — bit-equivalence preserved

Run the full 129 C++ baseline diff. Expected:
- 129/129 traces match within FP tolerance (~1e-12) under the natural pairing.
- The new filter in `diff_gmstab_dumps.py` correctly skips the trailing row at sites
  where the algorithm would now emit one extra observation.
- Cross-pairing combos (if any) in the sweep show correct exit semantics — `reason
  = ATOL` reachable, no spurious `DIVERGED_ITS`.

### 5.3 Manual smoke test — corner cases

```bash
# Exactly the configuration that failed before:
mpirun -n 1 ./tests/ex_gmstab_pcleft_jacobi -ksp_atol 1e-10 -ksp_max_it 100000
# Expected: reason=KSP_CONVERGED_ATOL, rnorm <= 1e-10, all 3 gates PASS

# Confirm natural pairings still cheap:
./tests/ex_gmstab_pc_sweep -ksp_pc_side right -ksp_norm_type unpreconditioned
# Expected: same iter counts as before; no extra MatMults.

# Confirm bit-equivalence on init prefix:
./tests/ex_gmstab_phase3a
# Expected: PASS (init prefix unchanged by this follow-up).
```

### 5.4 Code review — no semantics drift

- Confirm no new `ksp->normtype` reads added to algorithm code (Group A and B are
  pure deletions).
- Confirm comment updates accurately describe new behavior (especially the C++
  `solver.cpp:171-174` reference — it's now historical, not load-bearing).
- Confirm `KSPSetSupportedNorm` declarations unchanged
  (`gmstab.c:997-1002` should still list all four (norm, side) priorities).

---

## 6. Cold-start checklist

If picking this up with no recent context, the steps to get oriented are:

1. **Read `PC_LEFT_NORM_DISCUSSION.md`** end-to-end. The transcript explains the
   "why" in conversational form.
2. **Read this file (`PHASE4_FOLLOWUP_PLAN.md`) §2** to refresh the bug class and
   the chosen fix.
3. **Verify Phase 5a is done** by checking task #50 status. If it's still pending,
   stop — this follow-up is blocked.
4. **Read `tests/run_gmstab_tripwires.sh`** to understand the validation suite shape.
   Run it baseline (`SKIP_BUILD=1 ./run_gmstab_tripwires.sh`) and confirm 64/64.
5. **Open the six target sites** in your editor and read each one's surrounding
   ~30 lines:
   - `gmstab.c:280-340`
   - `gmstab_cycle1.c:108-130`
   - `gmstab_cycle2.c:150-165` and `gmstab_cycle2.c:470-485`
6. **Plan the diff.** §3.1 above prescribes each change; verify line numbers
   haven't drifted via `git blame`. If they have, adjust references and re-anchor
   on surrounding code patterns.
7. **Apply Group A first** (4 sites, post-Snapshot, pure cleanup). Run tripwires.
   They should still pass — Group A doesn't change behavior under any (side, normtype)
   pairing the natural-flow loop currently exits cleanly.
8. **Apply Group B** (2 sites, pre-Snapshot, drop entirely). Add filter from §3.2.
   Run tripwires + Phase 5a. Diff scripts should now show zero divergence.
9. **Optionally restore strict TOLABS** in the four PC tripwires that use GATE_TOL
   workaround (§3.2 last block). Verify they still pass.
10. **Update `PHASE3_STATUS.md`** with a Phase 4 follow-up section noting:
    - The cleanup is applied
    - The trace asymmetry under natural pairing is filtered by diff scripts
    - The cross-pairing stall is now correctly handled (no premature short-circuit)
    - User-visible behavior under PC_LEFT + UNPRECONDITIONED is now: longer iter
      count, but converges (or honestly fails on conditioning-ratio fundamentals)

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Group A change breaks bit-equivalence on validation paths I didn't anticipate | Run Phase 5a sweep before declaring done; the 129-baseline diff will catch any unexpected trace divergence |
| Group B + filter logic incorrectly drops a *real* mid-trace divergence | Filter explicitly checks `iter_norm < atol` AND `len(petsc) == len(cpp)+1` AND it's the *last* row. Three conditions — hard to misfire |
| `pcleft_jacobi` tripwire becomes too slow under strict 1e-10 (~50k iters) | Keep GATE_TOL workaround in that one test; document explicitly. Other tests (bjacobi/ilu/asm) should be unaffected |
| Future port additions (Phase 4c, 4e) re-introduce the `beta < abstol` pattern | Add a code-review checklist item: "no new inline `beta < ksp->abstol` short-circuits" |
| Comment about C++ `solver.cpp:171-174` becomes confusing if updated incorrectly | Rewrite the comment in §3.1 Group B to clearly state "we deliberately deviate from C++ here; trace asymmetry is filtered in tests/" |

---

## 7.5. Latent bug: `PC_RIGHT + KSP_NORM_PRECONDITIONED` reports wrong norm

**Discovered:** 2026-05-03, during user-facing discussion of cross-pairing costs.

**Symptom:** With `KSPSetPCSide(ksp, PC_RIGHT)` and `KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED)`,
`KSPConvergedDefault` is fed `iter_norm = gms->beta`, which under PC_RIGHT is
`||r_unprec||` (the algorithm tracks the unpreconditioned residual natively under
PC_RIGHT). The user asked for `||r_pre||` and got `||r_unprec||` silently. Their `atol`
is being applied to the wrong scalar.

**Diagnosis:** `gmstab_helpers.c:212` does `case KSP_NORM_PRECONDITIONED: rnorm_for_check = iter_norm;`
unconditionally. That's correct under PC_LEFT (where iter_norm IS the prec norm) and
wrong under PC_RIGHT (where iter_norm is the unprec norm).

**Registration** declares `(KSP_NORM_PRECONDITIONED, PC_RIGHT)` supported at priority 2
(`gmstab.c:998`), so the combo doesn't fail at setup — it just returns wrong results.

**Reference:** CG handles this correctly at `cg.c:170-172` — explicitly applies `B⁻¹`
to the residual when `KSP_NORM_PRECONDITIONED` is requested.

**Fix:** make the switch in `gmstab_helpers.c:209-217` PC-side-aware:

```c
switch (ksp->normtype) {
case KSP_NORM_UNPRECONDITIONED:
    rnorm_for_check = norm_true;       // already computed via MatMult above
    break;
case KSP_NORM_PRECONDITIONED:
    if (ksp->pc_side == PC_LEFT) {
        rnorm_for_check = iter_norm;   // algorithm-native — free
    } else {  /* PC_RIGHT or PC_SYMMETRIC */
        Vec r_pre_tmp;
        PC pc;
        PetscCall(KSPGetPC(ksp, &pc));
        PetscCall(VecDuplicate(b, &r_pre_tmp));
        PetscCall(PCApply(pc, r_true_vec, r_pre_tmp));
        PetscCall(VecNorm(r_pre_tmp, NORM_2, &rnorm_for_check));
        PetscCall(VecDestroy(&r_pre_tmp));
    }
    break;
case KSP_NORM_NONE:
    rnorm_for_check = 0.0;
    break;
default:
    SETERRQ(...);
}
```

This requires keeping the `r_true` vector from the `norm_true` block alive past line
190 — refactor to defer the `VecDestroy(&r_true)` until after the switch.

**Cost:** one extra `PCApply` per snapshot under `PC_RIGHT + KSP_NORM_PRECONDITIONED`.
Symmetric to the PC_LEFT + UNPREC cost but typically cheaper (PCApply ≤ MatMult for
most PC types; MUCH more for multigrid).

**Bundle with this follow-up?** Yes. The fix is small (~15 lines) and topically
identical (PC-side × normtype dispatch). Treat as part of the same patch series.
Validation: add a tripwire `tests/ex_gmstab_pcright_normpre.c` that runs PC_RIGHT +
NORM_PRECONDITIONED and verifies the reported rnorm matches an externally-computed
`||B⁻¹·(b - A·x)||`.

---

## 7.6. Final audit pass — additional findings (2026-05-03)

The final bug-fix audit pass before declaring Phase 4 done surfaced these items.
Acted on immediately (not deferred to this follow-up patch):

**Fixed in commit immediately following PHASE4_FOLLOWUP_PLAN draft:**

- **PC_SYMMETRIC silent-wrong-answer (gmstab.c:1001-1005, gmstab_helpers.c:50/68/163,
  gmstab_internal.h:82).** PC_SYMMETRIC was registered as supported across three
  norm types but the helper code (FinalizeSolution_Private and Snapshot_Private's
  unwrap branch) treated it as PC_RIGHT — using full `PCApply(B⁻¹)` for the unwrap
  rather than `PCApplySymmetricRight(B_R⁻¹)` that split-symmetric PCs need. Silent
  wrong-answer for any user setting PC_SYMMETRIC. **Fix: dropped PC_SYMMETRIC from
  KSPSetSupportedNorm registration.** PETSc setup now rejects PC_SYMMETRIC at solve
  time (matches GMRES convention). Helper-code branches at gmstab_helpers.c:68 and
  :163 become unreachable — left in place with comments updated to reference
  Phase 4c, not removed.

**Documented but deferred (no action needed):**

- **`gms->beta_pc` field declared in `gmstabimpl.h:117` but never assigned.** Dead
  code reserved for the (1.2 / option-(i)) follow-up implementation. When that lands,
  `beta_pc` is the natural place to track the "non-native" residual norm under the
  cross-pairing. Leaving in place as documented placeholder.

- **`KSP_NORM_NATURAL` not handled.** `gmstab_helpers.c:215` falls through to the
  `default: SETERRQ` case, which is correct behavior. Could be made more explicit
  (preflight check at solve start) but current handling is defensive — error message
  is clear enough.

- **CSV trace columns `iterres` / `trueres` are PC-side-dependent in meaning.** Under
  PC_LEFT, `iterres = ||r_pre||`; under PC_RIGHT, `iterres = ||r_unprec||`. Documented
  in `USER_MANUAL.md` §9 ("Diagnostics"). Diff scripts in `tests/` already handle
  this implicitly because they only compare against C++ baselines that ran in the
  same algebra.

- **Inner pGMRESm/augGMRESm could skip norm work under `KSP_NORM_NONE`.** Perf
  optimization, not correctness. Defer to Phase 6 (perf pass) if it ever becomes a
  hot path.

- **Restart heuristic `beta_curr < c_restart * beta_local` is algorithm-residual-based**
  and intentionally so — both betas are in algorithm-native algebra so the
  comparison is self-consistent regardless of user normtype choice. Documented in
  `gmstab.c` comments at lines 379–395; no fix needed.

---

## 8. Out-of-scope (do not bundle)

- **PC_SYMMETRIC support / Phase 4c** — separate concern, separate plan.
- **GPU PC validation / Phase 4e** — separate concern, separate plan.
- **Reducing the 4–6 snapshots/cycle baseline cost** — algorithmic deviation from C++
  reference; would break bit-equivalence intentionally. Larger scope, not this follow-up.
- **Changing `KSPSetSupportedNorm` declarations** — option (i) preserves all current
  (side, normtype) pairings as supported. No registration changes.
- **Adding `KSP_NORM_NATURAL` support** — gmstab doesn't support it currently and
  this follow-up doesn't add it. Separate concern if it ever comes up.

---

## 9. Appendix: site-by-site reference

The six inline check sites, with current code excerpts for cold-start orientation:

```c
// ──────────────────────────────────────────────────────────────────────
// Site 1: gmstab.c:287 (post-init duplicate path)
//   Group A — drop the if-block entirely.
// ──────────────────────────────────────────────────────────────────────
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));   // line 270
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) { ... }         // line 272

if (gms->beta <= ksp->abstol) {                                            // line 287 ← DROP
  ksp->reason = KSP_CONVERGED_ATOL;
  PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
  PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ──────────────────────────────────────────────────────────────────────
// Site 2: gmstab.c:308 (force_l1_only validation path)
//   Group A — drop beta-branch, simplify control flow.
// ──────────────────────────────────────────────────────────────────────
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));   // line 306
if (gms->beta <= ksp->abstol) {                                            // line 308 ← DROP
  ksp->reason = KSP_CONVERGED_ATOL;
} else if (!ksp->reason) {
  ksp->reason = KSP_DIVERGED_BREAKDOWN;
  PetscCall(PetscInfo(ksp, "...force_l1_only..."));
}

// ──────────────────────────────────────────────────────────────────────
// Site 3: gmstab.c:332 (force_l2_only validation path)
//   Group A — drop beta-branch, symmetric to site 2.
// ──────────────────────────────────────────────────────────────────────
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));   // line 330
if (gms->beta <= ksp->abstol) {                                            // line 332 ← DROP
  ksp->reason = KSP_CONVERGED_ATOL;
} else if (!ksp->reason) {
  ksp->reason = KSP_DIVERGED_BREAKDOWN;
  PetscCall(PetscInfo(ksp, "...force_l2_only..."));
}

// ──────────────────────────────────────────────────────────────────────
// Site 4: gmstab_cycle1.c:118 (pre-Snapshot, snapshot-skipping)
//   Group B — drop entirely; let next snapshot fire and gate exit on reason.
// ──────────────────────────────────────────────────────────────────────
/* C++ solver.cpp:171-174 ... skip mid-cycle snapshot in this branch ... */  // 112-117
if (beta < ksp->abstol) {                                                  // line 118 ← DROP
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));        // line 123
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) { ... }         // line 124

// ──────────────────────────────────────────────────────────────────────
// Site 5: gmstab_cycle2.c:158 (pre-Snapshot, snapshot-skipping)
//   Group B — drop entirely, symmetric to site 4.
// ──────────────────────────────────────────────────────────────────────
if (beta < ksp->abstol) {                                                  // line 158 ← DROP
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}
// ... cycle continues ...

// ──────────────────────────────────────────────────────────────────────
// Site 6: gmstab_cycle2.c:476 (post-Snapshot)
//   Group A — drop the beta-clause, restore symmetry with cycle1.c:124.
// ──────────────────────────────────────────────────────────────────────
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));        // line 475
if (beta <= ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)) {
//  ^^^^^^^^^^^^^^^^^^^^ ← DROP this clause                                  line 476
  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

Six sites, ~15 lines deleted, two comment updates. End state: every cycle exit goes
through `ksp->reason`, set by `KSPConvergedDefault` inside `Snapshot_Private`, which
honors `ksp->normtype` correctly under all (side, normtype) combos.
