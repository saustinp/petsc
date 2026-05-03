# Phase 3 — Status checkpoint

**Last updated:** 2026-05-02 (Phase 3b in progress)
**Branch:** `ksp-gmstab`

## Phases summary

| phase | status | notes |
|---|---|---|
| 0 — preflight + plan | ✅ committed `a3ecffcbd6d` + `b733e5911ca` | C++ oracle reproducible |
| 1 — KSP type skeleton | ✅ committed `20fcb15734b` | registration, options, public APIs |
| 2 — inner GMRES kernels | ✅ committed `aa821b3de64` | gmres_m, pgmres_m, aug_gmres_m + Givens, Arnoldi |
| 3a — Initialisation (port of Initialisation.m) | ✅ committed `1600f35eb9c` | **bit-equivalent on cdr_small at 3e-15** |
| 3a — small-dense + StabCoeffs | ✅ committed `77fae2e1fe9` | LAPACK helpers + 35° angle |
| 3b — GMstab1 cycle 1 | ✅ **VALIDATED** | bit-equivalent to C++ on cdr_small at 1.31e-12 drift (most rows 3e-15) |
| 3b — GMstab2 cycle 2 | ❌ not started | ~400 LOC, biggest cycle |
| 3b — driver loop | ❌ not started | flying-restart heuristic |

## Validated bit-equivalence baselines

The following validators live in `tests/`:

| validator | scope | status | acceptance criterion |
|---|---|---|---|
| `ex_gmstab_phase3a` | Initialisation on cdr_small (rows 0–1) | ✅ PASS | matvec exact, iterres/trueres drift ≤ 1e-10, **strict constructor row** |
| `ex_gmstab_init_nonzero_guess` | Constructor row with non-zero x0 | ✅ PASS | trueres == ‖b‖ within 1e-12 |
| `ex_gmstab_cycle1` | Cycle 1 force_l1 trace against C++ force_l1 trace | ✅ **PASS** | matvec exact, iterres/trueres drift ≤ 1e-10 (worst observed: 1.31e-12) |

Run all three with: `tests/run_gmstab_tripwires.sh`. Exit code is the
number of failing validators. SKIP_BUILD=1 skips the libpetsc rebuild
when iterating on a single validator.

## Phase 3b session — bugs found and fixed

While wiring cycle 1 into `KSPSolve_GMSTAB` we uncovered three real
bugs that were silently degrading the validation harness:

### Bug 1 — spurious `KSP_DIVERGED_DTOL` after Initialisation

**Symptom:** `ex_gmstab_phase3a` reported `KSPSolve` returning
reason=-4 (KSP_DIVERGED_DTOL) even though residuals matched the C++
oracle to machine epsilon. The Phase 3a tripwire accepted this as
"expected because cycle bodies aren't wired" but the underlying state
was wrong: any code path that *did* try to continue iterating (e.g.
the new force_l1_only path) would hit `if (ksp->reason)` and bail.

**Root cause:** `KSPGMSTABSnapshot_Private` was setting
`ksp->its = gms->snapshot_count` *after* incrementing
`snapshot_count`. So the very first call into PETSc's
`KSPConvergedDefault` was made with `n=1`, not `n=0`. The
`if (!n)` initialization branch in `KSPConvergedDefault` (which sets
`ksp->rnorm0` and `ksp->ttol`) was skipped, leaving `rnorm0=0`. The
DTOL check `rnorm >= divtol*rnorm0 = 0` then triggered on the first
non-zero residual.

**Fix:** capture `iter_idx = gms->snapshot_count` *before* the
increment and use that for both the CSV column and `ksp->its`. The
constructor snapshot now flows through `Snapshot_Private` (rather
than being written manually via `fprintf`) so `KSPConvergedDefault`
sees `n=0` first and properly initializes its rnorm0 / ttol.

### Bug 2 — missing matvec-counter increment in pgmres_m

**Symptom:** With cycle 1 wired in, the trace's matvec column froze
at the value reached at the start of the cycle, even though pgmres_m
was performing s additional matvecs internally.

**Root cause:** `gmstab_modgmres.c:595` — the `KSP_PCApplyBAorAB`
call inside `KSPGMSTABPGmresM_Private`'s outer loop did not increment
`*ws->matvec_count_ptr`. The other two matvec sites in the same file
(`KSPGMSTABGmresM_Private` line 417, `KSPGMSTABAugGmresM_Private`
line 733) did. Phase 3a never tripped this because Initialisation
only uses `gmres_m`.

**Fix:** added `if (ws->matvec_count_ptr) (*ws->matvec_count_ptr)++;`
on the line after the matvec call. Pattern now uniform across all
three inner kernels.

### Bug 3 — trueres regression for non-zero initial guess

**Symptom:** Found by audit, not by a failing tripwire (so a new
tripwire was added — see `ex_gmstab_init_nonzero_guess`).

**Root cause:** When I rerouted the constructor snapshot through
`Snapshot_Private` to fix Bug 1, the helper computes
`trueres = ||b - A*x_total||` via an uncounted MatMult. With the
validation harness using x0=0 this matches `||b||` accidentally, but
with a non-zero initial guess it reports `||b - A*x0||`. The C++
port's PerfMeasure constructor reports `||b||` *unconditionally*,
even with a non-zero x0. The original PETSc code had a comment about
this special case; my reroute removed the special handling.

**Fix:** `Snapshot_Private` now special-cases
`gms->snapshot_count == 0` (the constructor snapshot) and uses
`norm_true = iter_norm` directly, skipping the MatMult. Caller is
documented to pass `iter_norm = ||b||` for the constructor.

## Tripwires added this session

The audit motivated five additions, all in
`src/ksp/ksp/impls/gmstab/`:

| tripwire | location | catches |
|---|---|---|
| Strict constructor-row check | `tests/ex_gmstab_phase3a.c` | iter≠0, mv≠0, iterres≠trueres, non-zero runtimes on row 0 |
| Non-zero initial guess validator | `tests/ex_gmstab_init_nonzero_guess.c` | regression of the C++-port "trueres = ‖b‖ on constructor" semantics |
| Monotonic `matvec_count` invariant | `gmstab_helpers.c` (PetscCheck inside Snapshot_Private) | a missing-counter or double-counted matvec bug |
| Non-negative `snapshot_count` invariant | `gmstab_helpers.c` (PetscCheck inside Snapshot_Private) | accidental decrement |
| `_last_logged_matvec_count` cross-call audit field | `gmstabimpl.h` + reset in `KSPSolve_GMSTAB` | helps the above invariant catch drift across snapshots |
| Wrapper script | `tests/run_gmstab_tripwires.sh` | one-stop "make check" for Phase 3, exits non-zero on any failing validator |

## Cycle 1 validation — completed 2026-05-02

**Status:** Cycle 1 is bit-equivalent to the C++ reference port at
machine precision. Final `[cycle1] [PASS]` after closing two
structural gaps and proving 4 of the original HANDOFF "suspect" spots
were actually correct.

### How the bit-equivalence was demonstrated

The validation is now structured as three levels of evidence:

**Level 1 — per-snapshot CSV diff (ex_gmstab_cycle1)**

PETSc and the C++ port both run with `force_l1` enabled (always pick
cycle 1, skipping the natural cycle 2 default). Both emit the same
trace CSV format. We diff row-by-row.

| row | what | drift |
|---|---|---|
| 0 | constructor (mv=0, ‖b‖) | 2.22e-16 |
| 1 | post-Initialisation (mv=5) | 3.00e-15 |
| 2 | begin-cycle (mv=5, dup of post-Init) | 3.00e-15 |
| 3 | post-stab (mv=6) | 2.50e-15 |
| 4 | pgmres iter 0 (mv=7) | 2.55e-15 |
| 5 | pgmres iter 1 (mv=8) | 3.66e-15 |
| 6 | pgmres iter 2 (mv=9) | 3.39e-15 |
| 7 | pgmres iter 3 (mv=10) | 2.89e-15 |
| 8 | post-pgmres in cycle1 | 2.50e-15 |
| 9 | cycle-1 final (post-update) | **1.31e-12** |
| 10 | post-cycle in driver | **1.31e-12** |

The 1.31e-12 on rows 9–10 is the accumulated FP drift over the
post-pgmres update sequence (lq + trsm + gemv + outer-product BGS
of V0 against V1). At the input/output boundaries (rows 0, 1, 2, 3)
drift is single-digit ULPs.

**Level 2 — checkpoint-by-checkpoint matrix diff**

Both ports emit CSV dumps of every intermediate matrix/vector at 18
named checkpoints (chk00–chk17). The diff tool reports per-quantity
Frobenius diff and rel-Frobenius. 41/42 checkpoints match at machine
precision. The single non-match is `chk04 V0_postBGS`, and on
investigation it is *not* a bug:

> V0_in turns out to be in `span(V1)` to within rounding on cdr_small
> at the start of cycle 1. The block-GS subtraction reduces V0 columns
> 1, 2, 3 to vectors of magnitude ~1e-16 in arbitrary directions
> (their `||·||` is `2.1e-16`, `3.6e-17`, `2.3e-16` respectively, per
> the C[s+i, s+i] diagonal entries). Both ports then divide these
> noise-magnitude vectors by their tiny norms, producing unit vectors
> in arbitrary directions that differ between ports purely by
> accumulated FP rounding. **The downstream operations are insensitive
> to this** — `chk10 V0_postUpdate = [V1, V0_postBGS] * QfQz` matches
> between ports at 4e-14 because `QfQz` rotates the noise-direction
> contributions away. We confirmed this by checking `chk09 QfQz`
> (matches at 5e-14) and `chk10 V0_postUpdate` (matches at 4e-14)
> despite `chk04 V0_postBGS` differing at order 1.

This is a documented numerical edge case, not an algorithm bug.

**Level 3 — final residual sanity**

After cycle 1 on cdr_small with force_l1, both ports produce
β = 2.2572618199... — agreement to 1.3e-12 absolute.

### Bugs found and fixed during cycle1 validation

In addition to the three bugs caught in the earlier audit pass
(spurious DIVERGED_DTOL, missing matvec increment in pgmres_m,
trueres regression for non-zero initial guess), the cycle1 diff
process exposed two more structural gaps:

4. **Missing begin-of-cycle snapshot in PETSc driver.** The C++ main
   solver loop emits a snapshot at `solver.cpp:541-543` after Init
   returns, before entering the while loop — a duplicate of Init's
   final snapshot, with the same `mv` and `beta`. Our driver did not
   emit it. Fix: `KSPGMSTABSnapshot_Private` call at the top of the
   `force_l1_only` branch in `KSPSolve_GMSTAB`.

5. **Missing post-cycle snapshot in PETSc driver.** The C++ main
   solver loop emits a second snapshot at `solver.cpp:599-602` after
   each cycle returns — a duplicate of the cycle's own final snapshot.
   Fix: `KSPGMSTABSnapshot_Private` call at the bottom of the
   `force_l1_only` branch.

### Diagnostic-dump infrastructure (kept for cycle 2)

Two header-only helpers were added so the per-checkpoint diff tool
can be re-used for cycle 2:

| file | role |
|---|---|
| `gmstab_dump.h` (PETSc) | static-inline `KSPGMSTABDump{Mat,Vec,Dense,Vector1d,Real}_Private` |
| `gmstab_cpp/include/gmstab/dump.hpp` | header-only Eigen `dump_{mat,vec,real}` |
| `tests/diff_gmstab_dumps.py` | per-quantity Frobenius diff, sign-flip detection, shape checks |

All dump calls are gated on `GMSTAB_DUMP_DIR` env var; with it unset,
they're a single `getenv()` no-op per call. Cycle 2 will get the
same instrumentation pattern.

The C++ side honours `GMSTAB_FORCE_L1=1` to make the natural-flow
solver always pick cycle 1; this lets us validate cycle 1 in
isolation. A matching `GMSTAB_FORCE_L2` should be added for cycle 2
validation.

### Reference baseline

A new fixture was added at:

```
/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/cpp_residuals_force_l1.csv
```

— the C++ port's residuals trace under `GMSTAB_FORCE_L1=1`. This is
what `ex_gmstab_cycle1` diffs against. The natural-flow
`cpp_residuals.csv` baseline remains untouched and is what
Phase 5a's full validation will use after cycle 2 + driver are in.

## Second audit pass (post-cycle1 validation)

Per "no room for error" instruction, a second audit pass was performed
after cycle 1 was declared bit-equivalent. Findings:

### Issues found and fixed

**A. Parallel-correctness in dump helpers (FIRST PASS WAS A BAND-AID).**

First pass added `PetscCheck(comm_size == 1, ...)` to abort cleanly on
a parallel object. That was a band-aid: it prevented silent corruption
but didn't make the helpers actually work in parallel, and it failed to
notice that the cycle-1 W dump (which is on a *distributed* MatDense)
was using `KSPGMSTABDumpDense_Private` — the *replicated*-host-array
path — and silently writing only rank 0's local rows in parallel mode.
The sequential dump worked because the local array equals the global
view; parallel runs were emitting garbage past the local rank's range.

This was uncovered after the user pushed back on band-aid framing and
demanded full parallel correctness. Real fix:

1. Rewrote `KSPGMSTABDumpVec_Private` and `KSPGMSTABDumpMat_Private`
   to gather the parallel object onto rank 0 via
   `VecScatterCreateToZero` (column-by-column for matrices). Rank 0
   writes a single CSV; other ranks no-op. The output file is identical
   regardless of MPI rank count modulo floating-point rounding.
2. Added `KSPGMSTABDumpMatCols_Private(M, ncols)` for the over-allocated
   case (`ws->W` has m_max+1 columns; cycle1 only fills s+1).
3. Switched cycle1's W dump from the (wrong) DumpDense path to the
   (correct) DumpMatCols path.
4. Marked `Dump{Real,Vector1d,Dense}` as "replicated-on-every-rank
   helpers" — these are for small-dense PetscScalar arrays computed via
   global reductions and have identical content on every rank, so only
   rank 0 writes.
5. Verified the dump infrastructure with a sanity test: a 12×3 parallel
   MatDense with `M[i,j] = 1000*i + j` produces identical output files
   on 1, 2, and 4 ranks.

**B. Reproducibility verified.**
The full tripwire suite was run twice in succession with `SKIP_BUILD=1`
and produced identical 3/3 PASS each time. No stochastic state in the
diagnostic path.

**C. No-op gating verified.**
With `GMSTAB_DUMP_DIR` *unset* the cycle1 validator still PASSes (no
files written, no I/O attempted). The `KSPGMSTABDumpEnabled_Private`
guard correctly short-circuits each helper.

### Known latent issues (NOT introduced by this work)

These pre-date Phase 3b and are surfaced here for awareness:

**D. Repeated `KSPSolve` calls leak `gms->b_local`, `gms->x_global`,
`gms->r`.** `KSPSolve_GMSTAB` always allocates fresh vecs at lines
116–118 without first destroying any prior copies. PETSc only calls
`KSPReset_GMSTAB` (which destroys these) when the user explicitly
asks or when the operator changes. Two back-to-back `KSPSolve`s on
the same KSP would leak.
**Status:** known, pre-existing. No validator currently exercises this
pattern. Should be fixed before Phase 4 (PC sides) where multi-solve
patterns are likely to appear in real tests.

### Latent issues in the diagnostic infrastructure (acceptable)

**E. C++ `gmstab1_call_count` is a static int that persists across
multiple `solve()` invocations within the same process.** Within one
process the counter only resets between processes. Our validation
harness always runs `gmstab_run` as a fresh process, so this is fine
in practice. **Documented as a known limitation** of the diagnostic
mode.

**F. Convergence-callback resets `ksp->reason`.** PETSc's
`KSPConvergedDefault` unconditionally sets `*reason =
KSP_CONVERGED_ITERATING` as its first action. If a snapshot fires
mid-cycle that catches convergence (sets reason to ATOL), and a later
snapshot in the same cycle fires with a different residual, the
callback can reset reason. Cycle1's internal logic guards against
this via the `inner_terminated` flag and explicit `if (ksp->reason &&
ksp->reason != ITERATING)` checks at every potential exit point.
**Status:** safe for current usage; could matter under unusual PC
configurations (Phase 4).

### Untested edge cases (deferred to Phase 5a)

Cycle 1 has been validated only at `s = 4` on `cdr_small`. Phase 5a's
124-case sweep + 5 large baselines covers `s ∈ {1, 2, 4, 8, 16}` and
multiple matrix structures. The expected outcome is bit-equivalence
across all of them, but until run, edge cases like `s = 1` (where
the V0-against-V0 inner loop never executes) and very small N (where
the inner-pgmres workspace might allocate degenerate dimensions) are
not yet covered.

### V0_postBGS rank-deficiency noise — full explanation

This is the only checkpoint in the per-checkpoint diff that does NOT
match between sequential and parallel runs (or between PETSc and the
C++ port at exact bit level). The divergence is structural, not a
bug. This section captures the complete reasoning for the next
reader.

**The geometry (precise statement).** Initialisation produces
V0 = W[:, 0..s-1] · (Rh⁻¹ · Qz) and V1 = W[:, 0..s] · (Qh · Qz),
where W is the (s+1)-column Krylov basis from the inner GMRES.
Both V0 and V1 are s-dim subspaces of the same (s+1)-dim Krylov
space — *not* the same s-dim subspace. By dimension counting:
- range(V0) ∩ range(V1) is at least (s−1)-dim
- range(V0) has at most 1 dimension outside span(V1)
- range(V0) + range(V1) = the full (s+1)-dim Krylov space

**What the BGS does.** Cycle 1 orthogonalizes V0 against V1
(then against V0[:, 0..i-1]) and unconditionally normalizes:
```
C(s+i, s+i) = norm(V0[:, i])
V0[:, i]    = V0[:, i] / C(s+i, s+i)
```
This is verbatim from `GMstab1.m` lines 14-29 and is replicated 1:1
in `gmstab_cpp/src/solver.cpp:142-156` and our `gmstab_cycle1.c`.
The published algorithm does NOT do rank-deficiency detection.

**Numerical evidence on cdr_small (s=4).** The C diagonal records
post-orthogonalization norms:
- C[s+0, s+0] = 9.97e-3 (real residual — the one fresh direction)
- C[s+1, s+1] = 2.1e-16 (essentially zero — V0[:, 1] ⊂ span(V1, V0[:, 0]))
- C[s+2, s+2] = 3.6e-17 (essentially zero)
- C[s+3, s+3] = 2.3e-16 (essentially zero)

**What happens at the noise scale.** Dividing a 1e-16-magnitude
vector by a 1e-16 norm produces a unit vector pointing in whatever
direction the rounding error happened to fall. Different floating-
point reduction orders (sequential vs MPI-parallel; Eigen vs LAPACK
BLAS) yield different rounding patterns → different noise-direction
unit vectors. That is why V0_postBGS dumps disagree at order 1
between rank counts and across ports.

**Why it doesn't propagate.** The very next step builds
F = (1/τ)·C[:, s..2s-1] − C[:, 0..s-1]. Columns of F whose
contribution from the C-diagonal block is at noise level produce
near-singular columns. QR(F) extracts Qf such that the rows of Qf
paired with those rank-deficient F columns are themselves at
machine-epsilon magnitude. Then `V0_postUpdate := [V1, V0_postBGS] · QfQz`
multiplies noise unit vectors by noise-magnitude QfQz entries:

    O(1) noise direction × O(1e-16) QfQz entry = O(1e-16)

— completely swamped by the legitimate V1 contributions which are
O(1) × O(1) = O(1). The result is bit-equivalent across ports and
rank counts at machine precision (`chk10 V0_postUpdate` matches at
4.4e-14 between sequential and 4-rank parallel).

**Lifecycle of the noise:**
```
chk04 V0_postBGS: noise unit vectors live here  (cols 1..3)
                  ↓ build F = (1/τ)·C[:, s:2s] − C[:, 0:s]
chk05 F:          F has noise-magnitude columns (cols 1..3)
                  ↓ QR(F)
chk06 Qf, Rf:     Qf inherits the rank deficiency
                  ↓ combine with Qz from lq(-Z/Rf)
chk09 QfQz:       QfQz rows paired with noise V0 cols are ε-magnitude
                  ↓ V0_postUpdate := [V1, V0_postBGS] · QfQz
chk10 V0_postUpdate: noise × ε absorbed; matches at machine precision
```

So `V0_postBGS` *is* the intermediate state that gets overwritten at
chk10 by `V0_postUpdate`. The noise unit vectors live for two LAPACK
calls and two GEMMs — nothing downstream depends on their direction.

**Why the algorithm does this at all.** Letting QR(F) absorb the
rank deficiency is mechanically simpler than detecting "norm too
small → skip and zero" in the BGS. The mathematics is sound (a
rank-deficient matrix admits a valid QR factorization where the
trailing Q columns multiply trailing R rows that are exactly zero,
plus rounding) and the implementation is uniform. We faithfully
follow this published convention.

**Why this checkpoint is exempt from the bit-match gate.** Because
the noise direction is rounding-determined, no two implementations
will ever agree at this single checkpoint, and that is precisely
what the algorithm anticipates. The diff tool flags it; the
validator's pass/fail gate does not depend on it.

### Parallel-correctness validation (mandatory before cycle 2)

Cycle 1 was exercised on 2, 4, and 8 MPI ranks with cdr_small. All
runs completed with valid residuals matching the C++ sequential
reference within the parallel tolerance (1e-7):

| ranks | worst iter drift | worst true drift | result |
|---|---|---|---|
| 1 (sequential) | 1.31e-12 | 1.31e-12 | PASS |
| 2 | 9.87e-13 | 9.83e-13 | PASS |
| 4 | 2.05e-12 | 2.05e-12 | PASS |
| 8 | 2.90e-12 | 2.90e-12 | PASS |

Per-checkpoint diff between sequential and parallel dumps shows ALL 42
quantities matching at machine precision (worst non-noise drift 5e-13
on V0_final), with the documented exception of V0_postBGS (rank
deficiency, expected). `W^T W = I` invariant holds at 1e-15 across all
rank counts after the dump fix.

The tripwire suite now exercises this automatically:
`run_gmstab_tripwires.sh` runs the 3 sequential validators AND
`mpiexec -n {2,4,8} ex_gmstab_cycle1`, exiting non-zero on any
failure. Final state: **6/6 OVERALL PASS** (3 sequential + 3 parallel).

### Conclusion

After two audit passes, the cycle 1 work is parallel-correct in
addition to bit-equivalent at sequential machine precision.

| level | result |
|---|---|
| Sequential bit-equivalence with C++ port | ✅ 1.31e-12 worst drift on cdr_small |
| Parallel correctness (1, 2, 4, 8 MPI ranks) | ✅ all PASS within 1e-7 tolerance |
| Per-checkpoint dump diffs (seq vs parallel) | ✅ machine precision (5e-13) modulo documented V0_postBGS noise |
| W orthonormality `||W^T W - I||_F` | ✅ 1e-15 in all rank counts |

**Total bugs found and fixed across this Phase 3b session: 7.**
Three were uncovered only because the user demanded a more thorough
audit pass after each declaration of completeness:

1. Spurious `KSP_DIVERGED_DTOL` (off-by-one in `ksp->its`)
2. Missing matvec increment in `KSPGMSTABPGmresM_Private`
3. `trueres` regression for non-zero initial guess
4. Missing begin-of-cycle snapshot in driver
5. Missing post-cycle snapshot in driver
6. Latent parallel-safety in dump helpers (caught in audit pass 1)
7. **Cycle1 W dump using replicated-array path on a distributed MatDense
   (caught in audit pass 2 after parallel testing was demanded)**

**Cycle 1 is bit-equivalent to the C++ reference at sequential machine
precision AND parallel-correct on multiple MPI rank counts.** Tripwire
suite passes 6/6, reproducibly.

## Open Phase 3b work

Cycle 2 (`gmstab2` in C++ at `solver.cpp:246-457`, ~210 LOC) and the
flying-restart driver (`solver.cpp:553-602`) are next. Same validation
recipe as cycle 1:

1. Implement `KSPGMSTABCycle2_Private` as a 1:1 port of `gmstab2`.
2. Add a `force_l2_only` knob symmetric to `force_l1_only`.
3. Add matched checkpoint dumps in both ports.
4. Build a `cpp_residuals_force_l2.csv` reference fixture.
5. Add an `ex_gmstab_cycle2` validator that diffs PETSc's force_l2
   trace against the reference.
6. Once both cycles pass in isolation, wire them into the
   flying-restart driver and validate the full natural flow against
   the original `cpp_residuals.csv` (and then the 124 sweep cases).

## How to resume

```bash
# 1. Confirm we're on the validated baseline
cd /home/sam/hpc_stack/petsc
git status   # should be clean on ksp-gmstab
src/ksp/ksp/impls/gmstab/tests/run_gmstab_tripwires.sh
# Expect: 3/3 PASS

# 2. Generate the cycle 1 dump set (re-confirms checkpoint-level bit-equivalence)
rm -f /tmp/gmstab_dump/*.csv
GMSTAB_DUMP_DIR=/tmp/gmstab_dump GMSTAB_FORCE_L1=1 \
    /home/sam/hpc_stack/gmstab_cpp/build/gmstab_run \
    --linsys .../cdr_small/linsys.bin --P_bin .../cdr_small/P.bin \
    --s 4 --tol 1e-10 --maxmatvec 500 --maxruntime 30 \
    --out /tmp/cpp_run.csv >/dev/null
GMSTAB_DUMP_DIR=/tmp/gmstab_dump /tmp/ex_gmstab_cycle1 >/dev/null
python3 src/ksp/ksp/impls/gmstab/tests/diff_gmstab_dumps.py /tmp/gmstab_dump
# Expect: 41 ok, 1 DRIFT (chk04 V0_postBGS — documented noise)

# 3. Begin cycle 2 implementation per the recipe above.
```
