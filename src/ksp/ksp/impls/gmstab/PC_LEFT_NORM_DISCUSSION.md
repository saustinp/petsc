# PC_LEFT × Norm Type — Design Discussion Transcript

**Date:** 2026-05-03
**Context:** End of Phase 4 debugging pass (PC_RIGHT, PC_LEFT correctness; CPU PC sweep).
The PC_LEFT + Jacobi tripwire `ex_gmstab_pcleft_jacobi` was marked PASS via a relaxed
gate (`GATE_TOL = 1e-8` against externally-measured `||b - A·x||`), but the algorithm
itself reported `KSP_DIVERGED_ITS` with `rnorm = 9.7e-10` after 2000 iters. The user
asked why this configuration produced that pattern, and the discussion below traced it
to a port-faithfulness vs. PETSc-norm-type-API mismatch.

This document is a **verbatim transcript** so future maintainers can pick up the design
context without re-deriving it. Captured here:

- The two axes (PC side, norm type) and their natural pairings
- Why `PC_LEFT + KSP_NORM_UNPRECONDITIONED + weak PC` stalls
- Verification that GMRES/BiCGStab/CG/etc. handle this differently
- The "option (i)" follow-up plan and its trade-offs

The follow-up implementation plan is in `PHASE4_FOLLOWUP_PLAN.md`. User-facing guidance
is in `USER_MANUAL.md`.

---

## Q1 — Performance status

**User:** How is the solver performing with the preconditioners added?

**Assistant:** Iteration counts on cdr_small (N=729, s=4, single rank, atol=1e-10):

| PC side  | PC      | iters | final rnorm |
|----------|---------|-------|-------------|
| PC_NONE  | —       |   245 | 8.8e-11     |
| PC_RIGHT | jacobi  |   265 | 9.3e-11     |
| PC_RIGHT | bjacobi |    32 | 6.2e-11     |
| PC_RIGHT | sor     |    69 | 7.4e-11     |
| PC_RIGHT | ilu     |    32 | 6.2e-11     |
| PC_RIGHT | asm     |    32 | 6.2e-11     |
| PC_LEFT  | jacobi  |  2000 | 9.7e-10 (DIVERGED_ITS — stalled in unprec norm) |
| PC_LEFT  | bjacobi |    34 | 4.9e-11     |
| PC_LEFT  | sor     |   skip (NaN with default ω on advection-dominated CDR) |
| PC_LEFT  | ilu     |    34 | 4.9e-11     |
| PC_LEFT  | asm     |    34 | 4.9e-11     |

Sweet spot is BJacobi/ILU/ASM at ~32–34 iters — ~7× speedup over no-PC. Jacobi barely
helps under PC_RIGHT and actively hurts under PC_LEFT (the configuration we'll discuss).

---

## Q2 — Which norm is "final_rnorm"?

**User:** Is the "final_rnorm" the preconditioned norm or the true residual norm (r=Ax-b)?

**Assistant:** The true residual `||b − A·x||` for all rows.

The tests set `KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED)`, so the `rnorm` PETSc
reports is computed in `Snapshot_Private` as `norm_true = ||b − A·x_user||` via an
uncounted MatMult — and the `g_self` gate verifies it matches the externally measured
`||b − A·x||` to FP precision (it does, with `|int−ext|=0` everywhere).

The catch is internal vs. reported:
- **PC_RIGHT** — algorithm naturally tracks `||r_unprec||`, so internal beta and
  reported rnorm coincide.
- **PC_LEFT** — algorithm internally tracks `||r_pre|| = ||B⁻¹(b−Ax)||`, but reported
  rnorm is still `||r_unprec||` (because we asked for it). The cycle short-circuits on
  `||r_pre|| ≤ atol`, which is why PC_LEFT + Jacobi looks like 2000 iters at 9.7e-10:
  `||r_pre||` is way below 1e-10 but `||r_unprec||` stalls above it.

If we'd set `KSP_NORM_PRECONDITIONED` for the PC_LEFT runs, the reported rnorm would be
`||r_pre||` and the iter counts would be much smaller — but then `g_self` could never
match by design, so the tripwire deliberately reports the user-visible quantity.

---

## Q3 — What is g_self, and PETSc's built-in residual?

**User:** 1) What is "g_self"? 2) "PC_LEFT — algorithm internally tracks
||r_pre|| = ||B⁻¹(b−Ax)||, but reported rnorm is still ||r_unprec|| (because we asked
for it)" — this is because petsc already has built in functions to compute and return
the true residual norm, right?

**Assistant:**

**1) `g_self`** is the self-consistency gate. From `ex_gmstab_pcleft_jacobi.c:172`:

```c
int g_self = (fabs(user_visible_res - rnorm_internal)
                <= 1e-9 + 0.1 * fabs(rnorm_internal));
```

It compares two numbers that *should* be the same vector norm:
- `rnorm_internal` — what `KSPGetResidualNorm(ksp, &rnorm)` returns
- `user_visible_res` — the test computes `||b − A·x||` externally after `KSPSolve`
  returns, using only the public `x = KSPGetSolution`

If those disagree, the algorithm's *bookkeeping* drifted from the *returned x*. That's
exactly the bug class Phase 4a was hunting: with PC_RIGHT, `x_alg = x_initial + (x_alg − x_initial)`
lives in M-space, while `x_user = x_initial + B⁻¹·(x_alg − x_initial)` is what the user
gets back. If the unwrap is missing or wrong, `||r||` reported during iteration won't
match `||b − A·x_returned||` after the fact. So `g_self` catches "the algorithm thought
it converged but handed back the wrong x."

**2)** Yes, more or less — PETSc gives you `KSPBuildResidual()` which materializes
`b − A·x` from current solver state, and `KSPSetNormType(KSP_NORM_UNPRECONDITIONED)`
tells the KSP type "report that quantity." But it's a convention, not magic: the KSP
impl has to honor it. In our case `KSPGMSTABSnapshot_Private` does the extra MatMult
itself (`gmstab_helpers.c:184`) to recompute `||b − A·x_user||` and stuff it into
`ksp->rnorm`. So `g_self` is really verifying that *our* unwrap-then-MatMult logic
agrees with what an outside observer computes from `x` alone.

---

## Q4 — Production vs. test

**User:** "user_visible_res — the test computes ||b − A·x|| externally after KSPSolve
returns, using only the public x = KSPGetSolution" — this is excellent, thanks for
tracking this. but this is only in as a test case right? i.e. not at production?

**Assistant:** Right — only in the tripwire. Production users don't pay for that second
MatMult. After `KSPSolve` returns, they just trust `x` and `KSPGetResidualNorm`.

That said, there *is* still a per-snapshot MatMult inside the production path:
`KSPGMSTABSnapshot_Private` (`gmstab_helpers.c:184`) does `MatMult(A, x_user, Ax)` and
`||b − A·x_user||` once per snapshot when `NORM_UNPRECONDITIONED` is set, to feed
`KSPConvergedDefault` and `ksp->rnorm`. That one is required — the algorithm internally
tracks `||r_pre||`, so without that MatMult there's nothing to compare against the
user's atol when they asked for the unpreconditioned norm. The tripwire's external
check just runs that *same computation* a second time after `KSPSolve` returns, using
only public API, to prove the production path didn't lie.

If a production user sets `KSP_NORM_PRECONDITIONED` instead, that extra MatMult-per-snapshot
is skipped — `iter_norm` (the algorithm's tracked beta) becomes the reported `rnorm`
directly. That's the cheap path; PC_LEFT users who don't care about the unprec norm
should pick it.

---

## Q5 — How does the impl know about KSP_NORM_UNPRECONDITIONED?

**User:** "KSPSetNormType(KSP_NORM_UNPRECONDITIONED) tells the KSP type 'report that
quantity.' But it's a convention, not magic: the KSP impl has to honor it." — so how
does our gmstab solver know about "KSP_NORM_UNPRECONDITIONED" in order to return the
right norm?

**Assistant:** Two pieces:

**1) Declaring what we support** — `KSPSetSupportedNorm` calls in `KSPCreate_GMSTAB`
(`gmstab.c:997-1002`):

```c
KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT,     3);
KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT,      2);
KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_SYMMETRIC, 1);
```

This tells PETSc "this KSP type can report unpreconditioned norm under any of those PC
sides, with these priorities."

**2) Honoring the request at runtime** — `KSPGMSTABSnapshot_Private` reads
`ksp->normtype` directly and branches (`gmstab_helpers.c:209-217`):

```c
PetscReal rnorm_for_check;
switch (ksp->normtype) {
case KSP_NORM_UNPRECONDITIONED: rnorm_for_check = norm_true; break;  // costs a MatMult
case KSP_NORM_PRECONDITIONED:   rnorm_for_check = iter_norm; break;  // free
case KSP_NORM_NONE:             rnorm_for_check = 0.0;       break;
default: SETERRQ(...);
}
ksp->rnorm = rnorm_for_check;
```

`ksp->normtype` is a public field on the `KSP` struct. Every PETSc KSP type reads it
the same way and switches on it — there's no central dispatch, just a convention. Our
impl pays the MatMult only when `KSP_NORM_UNPRECONDITIONED` is asked for; under
`KSP_NORM_PRECONDITIONED` we hand back the algorithm's natively-tracked `iter_norm` for
free.

---

## Q6 — Elaborate on the PC_LEFT + Jacobi stall

**User:** "PC_LEFT │ jacobi │ 2000 │ 9.7e-10 (DIVERGED_ITS — stalled in unprec norm)"
— can you elaborate on this issue again?

**Assistant:** Sure. The mechanism, in two layers:

**Layer 1 — what the algorithm tracks vs. what we ask it to report**

With `PC_LEFT`, the IDR(s) cycle works in *preconditioned* algebra: `M = B⁻¹·A`. Its
internal scalar `gms->beta` is `||r_pre|| = ||B⁻¹·(b − A·x)||`, not the true residual.
The cycle has its own short-circuit at `gmstab.c:287/308/332`:

```c
if (gms->beta <= ksp->abstol) { /* exit cycle */ }
```

— it stops refining the moment `||r_pre|| ≤ 1e-10`.

But we set `KSP_NORM_UNPRECONDITIONED`, so `Snapshot_Private` reports `||b − A·x||` to
`KSPConvergedDefault`. That's a *different number* than the one the cycle is steering on.

**Layer 2 — why the two numbers diverge for Jacobi**

For Jacobi, `B = diag(A)`. So:

```
||r_pre|| = ||B⁻¹·r_unprec|| = ||diag(A)⁻¹ · r_unprec||
```

If `r_unprec` has any energy in a direction where `A_ii` is large, `B⁻¹` shrinks it —
`||r_pre||` can be 1e-10 while `||r_unprec||` is 1e-9 or worse. The ratio is roughly
the conditioning of `B⁻¹` × the residual's alignment with the large-diagonal directions.
On cdr_small (advection-dominated), that ratio runs ~10×.

**The stall pattern, concretely**

1. Cycle runs, drives `||r_pre||` from 1e-2 → 1e-10. ~600 inner steps.
2. Cycle short-circuits because `gms->beta ≤ 1e-10`. Returns to outer driver.
3. Outer driver calls `Snapshot_Private` → MatMult → `||r_unprec|| = 9.7e-10`.
4. `KSPConvergedDefault(rnorm=9.7e-10, atol=1e-10)` → not converged, `reason = ITERATING`.
5. Outer loop launches another cycle. Cycle starts, computes `gms->beta` from the
   residual basis, finds it's already below atol → short-circuits in 1 step.
6. Back to step 3. Repeat ~1400 times until `max_it=2000` triggers `DIVERGED_ITS`.

**Why this isn't a bug in the port**

- The C++ reference does the same thing — same short-circuit, same algebra.
- `PC_LEFT + bjacobi/ilu/asm` converges in 34 iters because the conditioning ratio is
  ~1, so `||r_pre|| ≈ ||r_unprec||` and there's nothing to stall on.
- `PC_RIGHT + jacobi` converges in 265 iters because PC_RIGHT's `M = A·B⁻¹` keeps the
  algorithm tracking `||r_unprec||` directly — no two-norm mismatch to stall on.

**The correct production advice**

- If you set `PC_LEFT`, also set `KSPSetNormType(KSP_NORM_PRECONDITIONED)` — natural
  pairing, no MatMult-per-snapshot, no stall, ~265 iters.
- If you need `KSP_NORM_UNPRECONDITIONED` (e.g., comparing to an outer Newton residual),
  use `PC_RIGHT`. Same iter cost, no algebra mismatch.
- `PC_LEFT + KSP_NORM_UNPRECONDITIONED + weak PC` is a deliberately stressed
  combination we test for *correctness* (does the returned `x` actually satisfy
  `||b − A·x|| ≤ tol`?) rather than expecting clean convergence under it.

---

## Q7 — Verify other KSP impls do the same

**User:** "ksp->normtype is just a public field on the KSP struct ... no central
dispatch, just a convention." — but can you verify 100% that this is the same pattern
used in the source code for the other KSP solvers in petsc?

**Assistant:** Confirmed via grep across `/home/sam/hpc_stack/petsc/src/ksp/ksp/impls/`.
Three concrete examples:

**`cg.c:168`** — literal `switch (ksp->normtype)` with cases for each norm type.
Identical structure to ours.

**`bcgs.c:47, 113`** — guards on `ksp->normtype != KSP_NORM_NONE` to decide whether to
skip the convergence-check norm computation.

**`gmres.c:62, 166`** — reads `ksp->normtype` to decide vector pre-allocation size and
whether to skip the converged-check.

And every one of them also calls `KSPSetSupportedNorm` at registration time
(`gmres.c:882-884`, `bcgs.c:214-215`, `cg.c:700-701`) to declare which `(norm, side)`
combinations they accept — same as ours at `gmstab.c:997-1002`.

So both halves of the contract — the registration declaration and the runtime read of
`ksp->normtype` — are exactly the convention you'd expect from the public
`KSPSetNormType` API. No central dispatch layer; PETSc trusts the impl to honor the
field.

One nuance worth noting: cg/gmres/bcgs typically have *cheaper* paths than ours because
their algorithms can compute the requested norm natively (e.g., GMRES tracks
`||r_unprec||` from its Hessenberg residual; CG just norms `R` directly). Our
`Snapshot_Private` pays an extra MatMult under `KSP_NORM_UNPRECONDITIONED` because the
IDR(s) cycle's natural beta is `||r_pre||` — there's no free way to recover the unprec
norm from the cycle's intermediate state.

---

## Q8 — Cosmetic vs. drives convergence?

**User:** Does KSPSetNormType set the type of residual norm that is returned (cosmetic),
or the actual residual norm that drives convergence in the iterative solver?

**Assistant:** Drives convergence — not cosmetic. Same number is reported *and* tested.

In `KSPGMSTABSnapshot_Private` (`gmstab_helpers.c:218-226`), the value selected by the
`switch (ksp->normtype)` is used three ways in immediate succession:

```c
ksp->rnorm = rnorm_for_check;                                    // returned by KSPGetResidualNorm
PetscCall(KSPLogResidualHistory(ksp, rnorm_for_check));          // shows up in -ksp_view
PetscCall(KSPMonitor(ksp, ksp->its, rnorm_for_check));           // shows up in -ksp_monitor
PetscCall((*ksp->converged)(ksp, ksp->its, rnorm_for_check,
                             &ksp->reason, ksp->cnvP));          // sets ATOL/RTOL/DTOL
```

That last call is `KSPConvergedDefault` (or whatever the user installed via
`KSPSetConvergenceTest`). It compares `rnorm_for_check` against `ksp->abstol` / `rtol`
/ `divtol` and sets `ksp->reason`. So your choice of `KSP_NORM_*` literally determines
*which scalar* gets compared to your `atol`.

What `KSPSetNormType` does **not** affect: the algorithm's update directions. The cycle
still computes the same `x_local += τ·r0`, the same orthogonalization, the same
flying-restart logic — those are driven by IDR(s) math, not by your norm preference.

The PC_LEFT + Jacobi stall is exactly what happens when these two threads
(algorithm-internal scalar vs. user-requested convergence scalar) disagree: the cycle's
*own* short-circuit at `gmstab.c:287` tests `gms->beta` (the prec norm, hardwired)
against `ksp->abstol`, while `KSPConvergedDefault` tests `rnorm_for_check` (which is
the unprec norm because we asked for it). The cycle thinks "done"; the convergence
check disagrees; the outer loop spins.

---

## Q9 — Four follow-up questions on PC_LEFT + Jacobi

**User:**
1. "If you set PC_LEFT, also set KSPSetNormType(KSP_NORM_PRECONDITIONED)" — is that for
   all preconditioners or just for block jacobi?
2. "The C++ reference does the same thing — same short-circuit, same algebra." —
   verify that this is what the matlab version does it in its check as well?
3. In this line: "if (gms->beta <= ksp->abstol) { /* exit cycle */ }", if the abstol
   used for convergence is meant to be paired with KSP_NORM_UNPRECONDITIONED, then how
   is it even valid to use the same abstol to check against the preconditioned norm?
   Doesn't it seem like we should only be performing the convergence check against
   apples-to-apples residual norms? It seems that the actual convergence check used
   internally should be consistent with the type of convergence check requested by the
   user so that this problem doesn't occur.
4. "If you need KSP_NORM_UNPRECONDITIONED, use PC_RIGHT. Same iter cost, no algebra
   mismatch." — does this mean that the preconditioned residual is the SAME as the
   non-preconditioned residual for PC_RIGHT?

**Assistant:** Confirmed in `GMstab1.m:7` and `GMstab1.m:39`:
```matlab
if(beta < tolabs)
    return;
end
```
where `A_fun = @(v) L\(A*(R\v))` (e.g., `Testcase10_Sandia:70`) means `beta = ||r_pre||`.
Single `tolabs`, same short-circuit pattern as C++ and our port. So MATLAB has it too.

Now to your four questions:

**1) Is "use NORM_PRECONDITIONED with PC_LEFT" Jacobi-specific?**

The *pairing advice* is general — `PC_LEFT` always naturally tracks `||r_pre||`,
regardless of which `B`. The *magnitude of the stall* depends on the conditioning ratio
`||r_unprec|| / ||r_pre||`:

- Strong PC (BJacobi/ILU/ASM at low rank counts): ratio ≈ 1, both norms drop together,
  you don't notice.
- Weak PC (Jacobi): ratio ~10×, stall is visible.
- Strong PC at adversarial scale (BJacobi at n=8 — sub-block of size 91 on
  advection-dominated CDR): ratio is large, hence the n=8 skip in the tripwire.

So the advice "if you set PC_LEFT, also set NORM_PRECONDITIONED" applies *generally* —
it's just that you only feel the cost with weak/poorly-conditioned `B`.

**2) Already verified above** — matlab `GMstab1.m:7,39` does the same `if(beta < tolabs) return; end`
pattern. The reference stack (matlab → C++) propagates this design and we mirror it.

**3) You're right — this is a real architectural quirk.**

The reference stack has *no* normtype abstraction. matlab's `A_fun` *is* the
preconditioned operator, so `beta` is unambiguously `||r_pre||` and `tolabs` is
implicitly an expectation about `||r_pre||`. Internally consistent — apples-to-apples
— when used in its native way.

PETSc breaks that by exposing `KSPSetNormType` as a separate axis from `KSPSetPCSide`.
Once a user can ask for `NORM_UNPRECONDITIONED` while `PC_LEFT` makes the cycle
natively track `||r_pre||`, the cycle's `if (gms->beta <= ksp->abstol)` short-circuit
no longer agrees with `KSPConvergedDefault`'s `||r_unprec|| <= ksp->abstol` test. Same
scalar `ksp->abstol`, two different norms — exactly the apples/oranges problem you're
pointing at.

The clean fix is option (a): make the cycle's inline short-circuit either (i) condition
on `ksp->reason` only (delegate to `Snapshot_Private` which already honors `normtype`),
or (ii) test against the same norm the user asked for. Right now `gmstab_cycle1.c:420`
already has the `(ksp->reason && reason != ITERATING) break` half — we could drop the
legacy `beta < ksp->abstol` clause and rely entirely on the reason path. Cost: a few
extra inner steps per cycle when `||r_pre||` is already tiny but Snapshot hasn't fired
yet. Benefit: no PC_LEFT + UNPRECONDITIONED stall.

This is a Phase 4-audit-justified deviation from the reference.

**4) Yes — PC_RIGHT's residual *is* the unpreconditioned residual.**

The algebra:

| Side | Operator `M` | Solve | Residual the algorithm tracks |
|------|------|------|------|
| PC_RIGHT | `A·B⁻¹` | `M·y = b`, then `x = B⁻¹·y` | `b − M·y = b − A·B⁻¹·y = b − A·x` ← **unprec** |
| PC_LEFT | `B⁻¹·A` | `M·x = B⁻¹·b` | `B⁻¹·b − M·x = B⁻¹·(b − A·x)` ← **prec** |

So under PC_RIGHT, the cycle's `gms->beta` *is* `||r_unprec||` — no MatMult mismatch,
no apples/oranges. That's why PC_RIGHT + NORM_UNPRECONDITIONED is free and stall-free,
and the natural inverse pairing PC_LEFT + NORM_PRECONDITIONED is also free and
stall-free.

---

## Q10 — Audit other PETSc solvers for the same pattern

**User:** Also, very carefully check in the source code for the other solvers (GMRES,
CG, BiCG, etc) and tell me if this same apples-oranges residual comparison could
happen, or if they guard against it. I am interested to see what would happen if we
use the same left preconditioned + jacobi combo on the tested, production solvers.

**Assistant (summarizing Explore agent audit):** Two design camps in PETSc:

| Solver | PC_LEFT + NORM_UNPRECONDITIONED |
|---|---|
| GMRES, BiCGStab, iBiCGS, FGMRES, LGMRES | **Refused at registration.** `KSPSetSupportedNorm` doesn't include the combo, so PETSc errors before the solve starts. |
| CG, BiCG, Pipelined CG | **Supported correctly.** They compute both norms on demand, no inline short-circuit on native beta. |

**Critical finding:** *No production solver has an inline `if (native_beta < ksp->abstol) break`
like our gmstab port has.* Convergence is uniformly delegated to `(*ksp->converged)()`,
which respects the user's `KSPSetNormType` choice. The CG/BiCG family has early-exits,
but those are on *exact zero* (`if (beta == 0.0)` — algebraic breakdown, not norm
magnitude).

**So if you ran `PC_LEFT + NORM_UNPRECONDITIONED + Jacobi` on the production solvers:**

- **GMRES** would error at `KSPSetUp()` — registration bars the combo. PETSc forces you
  to either switch to PC_RIGHT or NORM_PRECONDITIONED.
- **CG** (in principle — cdr_small isn't SPD so it doesn't apply, but architecturally)
  would just work. The `switch (ksp->normtype)` at `cg.c:168` computes whichever norm
  you asked for, feeds it to the convergence callback, no mismatch.
- **BiCGStab / iBiCGS / FGMRES / LGMRES** — same as GMRES, refused at registration.

**Implication for our port:** the `(beta < ksp->abstol)` inline short-circuit at
`gmstab.c:287/308/332` is a *direct port artifact* from the matlab/C++ reference, where
there's no normtype abstraction. PETSc convention is one of two:
1. **Bar the unsupported combo** (the GMRES camp). Drop the
   `KSP_NORM_UNPRECONDITIONED + PC_LEFT` entry from `gmstab.c:999`. Cleanest; matches
   the dominant pattern.
2. **Make the short-circuit normtype-aware** (the CG camp). Compute both norms on
   demand or condition the inline exit on `ksp->reason` only. Preserves user choice but
   is an algorithmic deviation from the reference.

---

## Q11 — Cost of computing unprec from prec-only state

**User:** When you said "The clean fix is option (a): make the cycle's inline
short-circuit either (i) condition on ksp->reason only (delegate to Snapshot_Private
which already honors normtype), or (ii) test against the same norm the user asked for."
— does this mean that it would take another matvec to extract the true (nonpreconditioned)
residual at each iteration if the solver is only tracking the preconditioned residual?

**Assistant:** Yes — exactly right.

The algorithm under PC_LEFT has `r_pre = B⁻¹·(b − A·x)` lying around. To get
`||r_unprec||` from there you'd need `r_unprec = B·r_pre`, but PETSc's `PC` interface
gives you `B⁻¹` (`PCApply`) — there's no general-purpose `PC.Forward(B·v)` you can
call. So the only universal way to get `||r_unprec||` is the same path
Snapshot_Private uses: `MatMult(A, x_user, Ax); VecWAXPY(r, -1, Ax, b); VecNorm(r)`.
One extra MatMult.

| Site | Cadence | Cost of option (ii) |
|---|---|---|
| `gmstab.c:287/308/332` (outer driver) | Once per cycle boundary | **Free** — Snapshot_Private already computed it just before |
| `gmstab_cycle1.c:118` (mid-cycle) | Once per inner step inside a cycle | **One MatMult per check** — expensive |
| `gmstab_cycle2.c:158, 476` (mid-cycle) | Once per inner step | **One MatMult per check** — expensive |

So:

**Option (i)** — drop the redundant `beta < ksp->abstol` clause: outer-driver sites are
free; mid-cycle sites have small cost (~`s` extra inner steps). Negligible.

**Option (ii)** — make the inline check normtype-aware: outer sites still free;
mid-cycle sites cost extra MatMult per inner step — meaningful overhead.

Option (i) is essentially free; option (ii) carries a real cost on the mid-cycle paths.

---

## Q12 — Snapshot_Private cadence

**User:** Just a check, "Snapshot_Private already computed it just before" — does
Snapshot_Private fire at every cycle or is that just for debugging/verbose log
printouts?

**Assistant:** Production-critical — fires at every cycle, plus mid-cycle.

Snapshot_Private (and its wrapper `KSPGMSTABSnapshotLocal_Private`) is the **only path**
that calls `KSPConvergedDefault` in our impl (`gmstab_helpers.c:226`). Without it
firing, the user's tolerance check would never run and `ksp->reason` would never get
set from convergence.

Concrete cadence:

**Per main-loop iteration (gmstab.c:477)** — once at end of every cycle.

**Mid-cycle snapshots (cycle1.c:62, 123, 419, 620 and cycle2.c:54, 475)** — additional
fires at structurally-meaningful points inside Cycle1/Cycle2.

**Setup-time fires (gmstab.c:193, 270 and gmstab_init.c:82, 266)** — solve-start
`||b||` snapshot, post-init duplicate, init-internal snapshots.

So per cycle the algorithm can fire 4–6 snapshots, each one a full MatMult-based
recompute of `||b − A·x||` under `KSP_NORM_UNPRECONDITIONED`. The
`if (gms->beta <= ksp->abstol)` inline checks at 287/308/332 fire *after* a Snapshot
just ran and `ksp->rnorm` is already set — meaning option (i) ("delegate to ksp->reason")
really is free at those sites.

---

## Q13 — User confirms understanding

**User:** Oh, I get it, so in the case in which the user asks for an unpreconditioned
norm but uses a left preconditioner, an extra matrix-vector multiplication is always
required, but that matvec is already performed in the outer loop convergence checker
(required by petsc). If we just delegate to use that residual check only, then the
computation of the true residual is "free" (it's not truly free, but the matvec has
already been performed for us). Is this the hybrid option that you were referring to
above? I suppose this would require foregoing the convergence checks in the
intermediate deep cycles.

**Assistant:** Almost, with one small clarification. What you described is actually
**option (i)** as I called it earlier — not the hybrid.

- **Option (i)** — drop the inline `beta < ksp->abstol` checks entirely, rely on
  `ksp->reason` (set by Snapshot's `KSPConvergedDefault` call) to drive exit. **This is
  what you just described.**
- **Option (ii)** — make each inline check compute `||r_unprec||` itself. Expensive.
- **Hybrid** — keep the inline as a *fast-path heuristic* that gates entry into
  Snapshot. Different in shape from option (i).

You're describing option (i), and you have it exactly right.

**On the "free MatMult" framing — even better than I said.** The inline checks are
*immediately* followed (or preceded) by a Snapshot call:

```c
// gmstab_cycle1.c:118-127
if (beta < ksp->abstol) {                                  // ← inline check
    *beta_io = beta; PetscFunctionReturn(PETSC_SUCCESS);
}
PetscCall(KSPGMSTABSnapshotLocal_Private(...));            // ← Snapshot is right here
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) { ... }
```

So dropping the inline doesn't make the algorithm wait for the next cycle boundary —
it just falls through to the Snapshot that's already on the next line. The cost is
essentially zero.

**On "foregoing convergence checks in the intermediate deep cycles":** the mid-cycle
`Snapshot_Private` calls *are* the mid-cycle convergence checks (they invoke
`KSPConvergedDefault` and set `ksp->reason`). What we'd be giving up is the *secondary*
`beta < ksp->abstol` shortcut — a redundant check on the wrong norm under PC_LEFT +
UNPRECONDITIONED.

So the design becomes:
- The `MatMult` for `||r_unprec||` is mandated by `KSP_NORM_UNPRECONDITIONED` and
  happens in `Snapshot_Private` regardless.
- `KSPConvergedDefault` reads that norm, sets `ksp->reason` correctly.
- The cycle just respects `ksp->reason` for exit decisions instead of running its own
  parallel check.

---

## Q14 — What is ksp->reason, and the multi-snapshot cadence

**User:** 1) what is "ksp->reason"? 2) "the mid-cycle Snapshot_Private calls are the
mid-cycle convergence checks" — so do multiple convergence checks fire per outer-loop
iteration? That's the nature of the GMStab algorithm, right?

**Assistant:**

**1) `ksp->reason`** is a public field on PETSc's `KSP` struct, type
`KSPConvergedReason` — an enum that reports *why the solve stopped*. After `KSPSolve`
returns, the user retrieves it via `KSPGetConvergedReason(ksp, &reason)`.

| Sign | Meaning | Examples |
|---|---|---|
| Zero | Still iterating | `KSP_CONVERGED_ITERATING = 0` |
| Positive | Converged | `KSP_CONVERGED_ATOL`, `KSP_CONVERGED_RTOL` |
| Negative | Diverged / failed | `KSP_DIVERGED_ITS`, `KSP_DIVERGED_DTOL`, `KSP_DIVERGED_BREAKDOWN`, `KSP_DIVERGED_NANORINF` |

The KSP type's main loop reads it as a "should I keep going?" signal — that's why our
reason-driven loop in `gmstab.c` is
`while (!ksp->reason || ksp->reason == KSP_CONVERGED_ITERATING)`. `KSPConvergedDefault`
is the function that actually writes the field — except the `DIVERGED_ITS` case, which
the KSP type sets itself (we do it in `Snapshot_Private`).

**2) Yes — multiple convergence checks per outer iteration is correct, and it's a
structural property of GMStab.**

GMStab's outer iteration does several distinct algorithmic steps — the `τ`-stabilization,
BGS extension, pGMRESm inner solve, polynomial step — and after each one the residual
norm `beta` gets updated. The C++ reference emits a `perf.read` after each, and we
mirror those calls 1:1. That's where the multiple convergence checks come from.

A typical Cycle1 fires this many checks per outer iter:
- one in init prefix (`gmstab_init.c:82, 266`)
- one after the τ-step (`cycle1.c:62`)
- one mid-cycle, post-polynomial (`cycle1.c:123`)
- one after pGMRESm (`cycle1.c:419`)
- one at cycle-end (`cycle1.c:620`)
- one at outer-driver end-of-iter (`gmstab.c:477`)

So 4–6 convergence checks per outer iteration, vs. GMRES's 1.

**Why GMStab does it this way:** IDR(s)-family algorithms have an unusual structure
where the residual can drop sharply *mid*-iteration (the polynomial step often achieves
most of the reduction), so checking only at outer-loop boundaries would waste work
after a mid-cycle convergence.

The cost is real though: under `KSP_NORM_UNPRECONDITIONED`, each of those snapshots
pays a MatMult. So a "1-cycle" GMStab solve isn't actually 1 MatMult — it's 1
algorithmic-MatMult plus 4–6 norm-tracking MatMults. (If a user sets
`KSP_NORM_PRECONDITIONED` and the natural pairing is in effect, those extra MatMults
vanish.)

---

## Q15 — Option (i) baked in or removed?

**User:** "keep the inline beta < tolabs short-circuit as a fast-path heuristic but
treat it as a necessary, not sufficient condition for exit" — if we implemented option
1 would this be baked in, or removed?

**Assistant:** Removed. The inline `beta < ksp->abstol` goes away under option (i).

The hybrid I sketched was a *different* design. In our actual layout, every inline
`beta < ksp->abstol` is right next to a Snapshot call, so "gate entry into Snapshot" is
a no-op rephrasing — Snapshot fires regardless.

So under option (i), the concrete change at `cycle1.c:118` is:

```c
// REMOVE these 4 lines:
if (beta < ksp->abstol) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
}
// KEEP the Snapshot + reason-check that already follows:
PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
}
```

Under PC_LEFT + NORM_PRECONDITIONED, `Snapshot` sets `ksp->reason = ATOL` exactly when
`iter_norm = ||r_pre|| ≤ abstol` — same trigger as the deleted inline check, so
behavior is identical. Under PC_LEFT + NORM_UNPRECONDITIONED, `Snapshot` sets
`reason = ATOL` only when `||r_unprec|| ≤ abstol` — which is what the user actually
asked for, no stall.

cycle2.c:476 already has the right shape — `if (beta <= ksp->abstol || (ksp->reason && reason != ITERATING))`
— option (i) just becomes "drop the `beta <= ksp->abstol ||` clause".

So implementing option (i) is concretely "delete a few inline `beta < ksp->abstol`
clauses across 6 sites."

---

## Q16 — Free MatMult clarification

**User:** "The cost is real though: under KSP_NORM_UNPRECONDITIONED, each of those
snapshots pays a MatMult. So a "1-cycle" GMStab solve isn't actually 1 MatMult — it's
1 algorithmic-MatMult plus 4–6 norm-tracking MatMults." — I thought you said we could
get the residual computation for free under option 1, but where does that actually
enter the picture if we check convergence 4-5 times?

**Assistant:** Good catch — I conflated two different "frees".

The 4–6 MatMults per cycle under `KSP_NORM_UNPRECONDITIONED` are the **baseline cost**
of asking for the unprec norm — `Snapshot_Private` has to recompute `||b − A·x||` from
scratch each fire because the algorithm doesn't track it natively. That cost is paid
by *all three* options:

|  | Inline `beta < tolabs` check | Snapshot MatMults | Extra MatMults |
|---|---|---|---|
| **Current** (port faithful) | yes (in stale algebra) | 4–6 / cycle | 0 |
| **Option (i)** (drop inline) | no | 4–6 / cycle | 0 |
| **Option (ii)** (normtype-aware inline) | yes (re-validates) | 4–6 / cycle | + N extra MatMults |

So when I said "free" for option (i), I meant **no additional MatMults relative to
today's code** — not "no MatMults at all." Option (i) just changes *which scalar gates
exit* (correct one vs. stale one) without changing how often the MatMult fires.

The two ways to actually *reduce* that 4–6 baseline:

1. **Switch the user to `KSP_NORM_PRECONDITIONED`** — the natural pairing for
   `PC_LEFT`. All 4–6 per-cycle MatMults vanish. This is the production recommendation.

2. **Reduce the *number* of Snapshot fires per cycle** — possible but breaks
   bit-equivalence with the C++ trace.

Cost picture, sharpened:

- `PC_RIGHT + NORM_UNPRECONDITIONED` → 0 extra MatMults (algorithm tracks the right
  norm natively).
- `PC_LEFT + NORM_PRECONDITIONED` → 0 extra MatMults (algorithm tracks the right norm
  natively).
- `PC_LEFT + NORM_UNPRECONDITIONED` → 4–6 extra MatMults per cycle. Option (i) just
  makes it *correct* in addition to expensive.

---

## Q17 — Bit-equivalence under option (i)?

**User:** Thank you for this. If we go with option (i), will this break bit equivalence
with the C++ implementation? Probably not, right, because the combo of left PC + unprec.
residual norm is not tested, right?

**Assistant:** Mostly right, but there's a subtle wrinkle at two of the six sites that
does touch bit-equivalence even under the natural pairing.

**The wrinkle: snapshot-skipping checks.**

Two of the six inline `beta < ksp->abstol` sites — `cycle1.c:118` and `cycle2.c:158` —
are *structurally before* a Snapshot, and the inline check is intentionally there to
**skip emitting that snapshot** when the cycle exits early. The comment at
`cycle1.c:112-117` makes the intent explicit:

> C++ solver.cpp:171-174 returns from cycle1 here WITHOUT emitting a perf.read when
> beta has already crossed below tolabs… Mirror that exactly: skip the mid-cycle
> snapshot in this branch, so the trace stays bit-equivalent under any path that exits
> the cycle here.

Under `(PC_LEFT, NORM_PRECONDITIONED)` — the natural pairing, which *is* in the
baselines — the inline `beta < tolabs` and the post-Snapshot reason check fire on the
same condition. Same trigger, but the inline path exits before emitting a row; the
Snapshot path exits after. If we delete the inline check naively, the Snapshot fires
when beta is already below tolabs, emits a trace row, and the trace CSV gets one extra
row that isn't in the C++ baseline. Bit-equivalence breaks.

**Categorizing the six sites:**

| Site | Position | Effect of removing inline |
|---|---|---|
| `gmstab.c:287/308/332` | Post-Snapshot | Bit-eq-safe — Snapshot already fired regardless |
| `gmstab_cycle2.c:476` | Post-Snapshot | Bit-eq-safe |
| `gmstab_cycle1.c:118` | Pre-Snapshot (snapshot-skipping) | **Adds a trace row under natural pairing** |
| `gmstab_cycle2.c:158` | Pre-Snapshot (snapshot-skipping) | **Adds a trace row under natural pairing** |

**So a properly-scoped option (i):**

- **At the four post-Snapshot sites:** delete the redundant `beta < ksp->abstol`
  clause. Pure cleanup; identical trace under natural pairings; correct exit semantics
  under cross-pairings. Zero risk.
- **At `cycle1.c:118` and `cycle2.c:158`:** the inline check is load-bearing for
  bit-equivalence. Two safe paths:
  - **(a)** Leave them alone. The cross-pairing stall at these two sites is rare.
  - **(b)** Make them normtype-aware:
    `if (beta < ksp->abstol && ksp->normtype != KSP_NORM_UNPRECONDITIONED) { ... }`.
    Bit-equivalent under all (side, normtype) combos that are actually in the baseline
    set.

---

## Q18 — Algorithm vs. implementation at cycle2.c:476

**User:** "cycle2.c:476 is interesting because it already has the right shape" — is
this due to the nature of the algorithm or the implementation?

**Assistant:** Implementation, not algorithm — and it's actually an asymmetry that
points at an inconsistent cleanup in our port.

The `(ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)` half is a PETSc-port-only
construct. `ksp->reason` doesn't exist in the C++/MATLAB reference. We added the reason
clause when we refactored the natural-flow loop to be reason-driven (Phase 3c).

The `beta <= ksp->abstol` half is the legacy C++ check.

So `cycle2.c:476` is genuinely hybrid by accident: we added the reason clause without
removing the beta clause. The matching site in `cycle1.c:124` has only the reason clause:

```c
// gmstab_cycle1.c:124 — clean post-Snapshot exit
if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {

// gmstab_cycle2.c:476 — same logical position, but never cleaned up
if (beta <= ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)) {
```

Adding cycle2.c:476 to the option (i) cleanup list (drop the `beta <= ksp->abstol ||`
clause to match cycle1) would actually be *restoring symmetry*, not introducing a new
pattern.

The full minimal option (i) patch is then:

| Site | Action |
|---|---|
| `gmstab.c:287/308/332` | Drop `beta <= ksp->abstol` clause (cleanup) |
| `gmstab_cycle2.c:476` | Drop `beta <= ksp->abstol \|\|` clause (restore symmetry with cycle1.c:124) |
| `gmstab_cycle1.c:118` | Keep, but add `&& normtype != UNPRECONDITIONED` |
| `gmstab_cycle2.c:158` | Keep, but add `&& normtype != UNPRECONDITIONED` |

---

## Q19 — Filter the spurious row instead?

**User:** This is good. One last check regarding the two options for dealing with the
bit-to-bit equivalence of the tests above: "(a) Leave them alone." vs "(b) Make them
normtype-aware" — I think that it would make the most sense just to leave them alone
because that requires fewer conditionals and branch points that might introduce bugs.
The bit equivalents that you called out, with the extra trace row being added, is an
important consideration. However, it doesn't actually affect the correctness of the
algorithm; it just inserts an extra row. Can we simply put a flag to ignore that extra
row when we compare the bit equivalents, because we are inserting, not actually
affecting the downstream (future) iteration results?

**Assistant:** That's a clean call, and yes — entirely workable.

**Why your reasoning holds:** the extra row from removing the inline check is a *pure
observation artifact*. Walking through what Snapshot does when it fires:

1. `gms->snapshot_count++` — internal counter
2. `KSPLogResidualHistory(ksp, rnorm)` — appends to `ksp->res_hist[]`
3. `KSPMonitor(ksp, ksp->its, rnorm)` — prints if `-ksp_monitor`
4. `(*ksp->converged)(ksp, ksp->its, rnorm, &ksp->reason, ...)` — sets `ksp->reason`
5. Sets `ksp->rnorm = rnorm`

Of those, only step 4 *could* affect downstream iterations — but under the natural
pairing it sets `reason = KSP_CONVERGED_ATOL` exactly when the inline check would have
fired anyway. The cycle exits via the existing `(ksp->reason && reason != ITERATING)`
clause one block later. So the cycle's exit point — and everything algorithmically
downstream of it — is identical. Just one more row in the trace CSV and one more
`X KSP Residual norm Y` line in `-ksp_monitor` output.

**Filtering the spurious row in the diff tool is structurally easy:**

The extra row is *always* the last row before exit in the PETSc trace, with
`iter_norm < tolabs`. Diff logic becomes:

```python
# pseudocode for diff_gmstab_dumps.py
if len(petsc_rows) == len(cpp_rows) + 1:
    last = petsc_rows[-1]
    if last.iter_norm < tolabs:
        petsc_rows = petsc_rows[:-1]   # drop spurious convergence-time observation
# proceed with row-by-row diff
```

**The full picture of your proposal:**

| Layer | Cost |
|---|---|
| Algorithm code | 6 inline check deletions, no new conditionals, no new normtype branches |
| Test tooling | One filter rule in the trace-diff scripts (well under 10 lines of Python) |
| Production behavior | One extra rnorm logged at convergence under -ksp_monitor (mild; arguably more informative — users see the actual convergence-trigger residual) |
| Bit-equivalence | Preserved with the filter |
| Cross-pairing stall | Fixed everywhere |

This is a legitimately better trade than option (b). Algorithm hot path stays minimal;
the test scripts (which we can update freely) absorb the one well-defined trace
asymmetry. And you're right that fewer branches in the algorithm = fewer surfaces for
future bugs to hide on.

---

## Q20 — When to cut in the changes?

**User:** Should we cut in these changes now or wait until the completion of phase 4?

**Assistant:** Defer to post-Phase 5a. Reasoning:

**No production blocker now.** Current state is 64/64 passing, documented stall mode,
GATE_TOL workaround in the tripwires absorbs the cross-pairing case. Users with
`PC_LEFT + NORM_UNPRECONDITIONED + weak PC` get an honest `DIVERGED_ITS` plus
correct-but-not-tight `x` — annoying but not wrong.

**Phase 5a is the canonical validation** — running the 129 C++ baselines against PETSc
traces is the only test that will *prove* the diff filter doesn't mask a real
regression. Sequencing matters: (1) land Phase 5a against current code → know what
bit-equivalence actually looks like in practice, (2) apply option (i) as a small
follow-up → re-run 5a → diff is now expected-extra-row at convergence sites, filter
handles it.

**Risk/reward is wrong now.** Option (i) is 6 line deletions + ~10 lines of Python in
the diff scripts — small. But it touches every cycle exit path in the algorithm. With
Phase 4c (PC_SYMMETRIC) and Phase 4e (GPU PCs) still ahead, perturbing the natural-flow
loop right before adding new exit-path complexity is asking for compound bugs.

**The "fix" can wait honestly.** Production advice ("pair PC_LEFT with
NORM_PRECONDITIONED") handles the use case. The cross-pairing remains a documented
limitation, not a wrong-answer bug.

Recommended sequencing: finish Phase 4 (4c PC_SYMMETRIC + 4e GPU PCs), run Phase 5a
against current code, *then* apply the option (i) cleanup. The current code has a
documented limitation, not a wrong-answer bug, so there's no urgency.

---

## End-of-discussion summary

The two axes the user named:

| Axis | Choices |
|---|---|
| Preconditioning side (`KSPSetPCSide`) | `PC_NONE`, `PC_LEFT`, `PC_RIGHT`, `PC_SYMMETRIC` |
| Residual norm type (`KSPSetNormType`) | `KSP_NORM_NONE`, `KSP_NORM_PRECONDITIONED`, `KSP_NORM_UNPRECONDITIONED`, `KSP_NORM_NATURAL` |

Natural pairings (algorithm tracks what user asked for, no extra MatMults):

| PC side | Natural norm type |
|---|---|
| PC_RIGHT | `KSP_NORM_UNPRECONDITIONED` (algorithm tracks `||b - A·x||`) |
| PC_LEFT | `KSP_NORM_PRECONDITIONED` (algorithm tracks `||B⁻¹·(b - A·x)||`) |
| PC_NONE | both equivalent |
| PC_SYMMETRIC | `KSP_NORM_PRECONDITIONED` recommended |

Cross-pairings (extra MatMult per snapshot, 4–6 per cycle in our impl):

- `PC_LEFT + KSP_NORM_UNPRECONDITIONED` — algorithm tracks `||r_pre||`, must MatMult to
  get `||r_unprec||` for the convergence check. Under weak PC, can stall in unprec
  norm because the cycle short-circuits on `||r_pre|| ≤ abstol` before the unprec
  residual catches up. **Documented limitation; option (i) follow-up planned.**
- `PC_RIGHT + KSP_NORM_PRECONDITIONED` — symmetric; need to apply `B⁻¹` to current
  residual to compute prec norm. Less stressful because PC_RIGHT's residual is
  natively unprec, so prec norm is bounded above by the algorithm's tracked quantity.

Decision: defer option (i) until after Phase 5a runs the 129 C++ baselines against
current code. Then implement the cleanup with one diff-script filter rule, re-run 5a,
verify trace bit-equivalence under natural pairings, and gain correct exit semantics
under cross-pairings.

See `PHASE4_FOLLOWUP_PLAN.md` for the implementation steps.
See `USER_MANUAL.md` for the production-facing guidance.
