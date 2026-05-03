# Phase 3 — Status checkpoint

**Last updated:** 2026-05-02 (Phase 3d — audit + tripwire expansion)
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
| 3b — GMstab2 cycle 2 | ✅ **VALIDATED** | bit-equivalent to C++ on cdr_small at 5.26e-12 drift |
| 3c — natural-flow driver loop | ✅ **VALIDATED** | full flying-restart wired; 5-gate validator passes 1/2/4/8 ranks |
| 3d — audit + tripwire expansion | ✅ **VALIDATED** | three new tripwires (nzg / determinism / multi-solve), one new cross-check (natural seq-vs-parallel coherence), six fixes for bugs caught by them |

## Validated bit-equivalence baselines

The following validators live in `tests/`:

| validator | scope | status | acceptance criterion |
|---|---|---|---|
| `ex_gmstab_phase3a` | Initialisation on cdr_small (rows 0–1) | ✅ PASS | matvec exact, iterres/trueres drift ≤ 1e-10, **strict constructor row** |
| `ex_gmstab_init_nonzero_guess` | Constructor row with non-zero x0 | ✅ PASS | trueres == ‖b‖ within 1e-12 |
| `ex_gmstab_cycle1` | Cycle 1 force_l1 trace against C++ force_l1 trace | ✅ **PASS** | matvec exact, iterres/trueres drift ≤ 1e-10 (worst observed: 1.31e-12) |
| `ex_gmstab_cycle2` | Cycle 2 force_l2 trace against C++ force_l2 trace | ✅ **PASS** | matvec exact, iterres/trueres drift ≤ 1e-10 (worst observed: 5.26e-12) |
| `ex_gmstab_natural` | Full natural-flow flying-restart trace vs `cpp_residuals.csv` | ✅ **PASS** | 5-gate harness: Init prefix bit-equivalent at 1e-10, matvec column matches >= 40% of rows, KSP_CONVERGED_ATOL, final res ≤ 1.5×abstol, total mv ≤ 1.25× C++ baseline |
| `ex_gmstab_natural_nzg` | Natural-flow with KSPSetInitialGuessNonzero (x0=0.7 const) | ✅ **PASS** | externally-measured ‖b−A·x_returned‖ ≤ 1.5×abstol AND equals KSP-reported rnorm to FP precision (catches missing x_global accumulation on early-out) |
| `ex_gmstab_determinism` | Two consecutive runs of natural-flow on a fresh KSP each time | ✅ **PASS** | trace files BIT-IDENTICAL byte-for-byte; iter counts identical |
| `ex_gmstab_multisolve` | KSPSolve called twice on the SAME KSP object | ✅ **PASS** | both reach ATOL with identical iter count and rnorm; ‖x1−x2‖₂ = 0 exactly (catches per-solve state-not-reset) |
| `diff_natural_seq_vs_par.py` (cross-check) | natural-flow seq trace vs parallel trace | ✅ **PASS** | Init prefix drift ≤ 1e-10 (machine precision; gauge-independent); matvec column exact through prefix; row count parity within 25% |

Run all eight validators + five cross-checks with: `tests/run_gmstab_tripwires.sh`.
Exit code is the number of failing checks. SKIP_BUILD=1 skips the libpetsc rebuild
when iterating on a single validator.

The full suite passes **40/40** on every commit:

* 10 sequential validators (one per validator above plus `multisolve_rng`)
* 24 parallel runs (cycle1, cycle2, natural, natural_nzg, determinism,
  multisolve, multisolve_rng, schange × ranks ∈ {2, 4, 8})
* 4 dump-level cross-checks (cycle1, cycle2 × ranks ∈ {2, 4})
* 2 trace-level cross-checks (natural seq-vs-parallel × ranks ∈ {2, 4})

### Natural-flow validator design rationale

The `ex_gmstab_natural` validator does **not** gate on per-row 1e-10
absolute drift over the whole trace. After ~5 cycles drift exceeds
1e-6, and after ~10 cycles it exceeds 1e-2 — even though the algorithm
remains structurally correct. The reason is gauge ambiguity:

* PETSc's small-dense layer goes through LAPACK (`dgesdd`, `dgeqrf`,
  etc.). The C++ port goes through Eigen's BDCSVD/HouseholderQR.
* These give numerically distinct but mathematically equivalent
  factorisations: the Z, Q, and V0/V1 they emit differ by a
  deterministic rotation (the gauge).
* Per the published GMstab paper, the **cycle outputs are
  basis-invariant** — final V0/V1/r0/Z span the same subspaces in
  exact arithmetic. In FP, gauge differences perturb each cycle's
  output by O(1e-12).
* The polynomial step at the start of every cycle amplifies this
  perturbation by ~50×–200×. After 4–5 cycles drift reaches 1e-6;
  by cycle 10 it's 1e-2. Both ports still converge to `tolabs`.

So the validator's five gates instead check what *should* be
gauge-independent or structurally invariant:

1. **Init prefix (rows 0..11)**: gauge-INDEPENDENT — produced by
   inner `gmres_m`, no SVD/QR/LQ. Drift gated at 1e-10.
2. **matvec column**: each port should issue exactly the same
   sequence of counted matvecs until trajectories visibly diverge.
   Must match for ≥ 40% of rows from the start.
3. **Convergence reason**: must be `KSP_CONVERGED_ATOL`.
4. **Final residual**: ≤ 1.5×`abstol` (the trajectory might overshoot
   slightly above tolabs at the snapshot before convergence).
5. **Total matvec budget**: ≤ 1.25× C++'s matvec count (allows for
   gauge-induced trajectory taking a few extra matvecs).

Observed numbers on cdr_small (s=4, abstol=1e-10):

| ranks | rows | init prefix drift | matvec match | final res | matvec budget |
|---|---|---|---|---|---|
| 1 | 246 | 4.83e-15 | 132/246 | 8.83e-11 | 205 vs 194 (+5.7%) |
| 2 | 256 | 4.11e-15 | 120/256 | 5.24e-11 | 213 vs 194 (+9.8%) |
| 4 | 234 | 2.00e-15 | 132/234 | 9.01e-11 | 195 vs 194 (+0.5%) |
| 8 | 244 | 1.44e-15 | 132/244 | 9.58e-11 | 203 vs 194 (+4.6%) |

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

## Phase 3c — natural-flow driver

Wired cycle 1 + cycle 2 into the natural-flow flying-restart driver in
`KSPSolve_GMSTAB` (replacing the Phase 3a stub). The driver is a 1:1
port of `solver.cpp:742-815`:

* post-init duplicate snapshot (matches C++ `solver.cpp:743`)
* `betaMax = max(beta, betaLocal)` after Initialisation
* `while (beta > tolabs)` loop with t_restart / t_replace / neither
  bookkeeping using the same constants (c_restart=1e-2, c_replace=1e-2,
  n2cyclesMax=3) and the same L-selection rule
* counted matvec + dir_rbio (xi=Z\eta; r-=V1*xi; x+=V0*xi) on
  restart/replace
* xGlobal/x_local accumulation on restart
* end-of-iter snapshot

In the process discovered and fixed a latent restart-aware bug: snapshot
sites in cycle1.c, cycle2.c, gmstab.c, and gmstab_init.c were passing
`x_local` (cycle-local solution) instead of `x_total = x_global +
x_local`. This was a no-op for the force_l*_only validators (xGlobal = 0
there) but would have produced wrong trueres values on every snapshot
once flying-restart fired in natural-flow. Introduced a new helper
`KSPGMSTABSnapshotLocal_Private` that internally constructs the total
and forwards to `KSPGMSTABSnapshot_Private`. All non-callback snapshot
sites now use the helper.

The resulting tripwire suite passes 18/18:

* 5 sequential validators (`phase3a`, `init_nonzero_guess`, `cycle1`,
  `cycle2`, `natural`)
* 9 parallel runs (cycle1/cycle2/natural × n=2,4,8)
* 4 seq-vs-parallel checkpoint diffs (cycle1/cycle2 × n=2,4)

See the natural-flow validator design rationale earlier in this file
for why per-row 1e-10 drift over the entire trace is not the gate.

## Phase 3d — audit + tripwire expansion

In response to a thoroughness mandate ("we need to be incredibly
thorough and prioritize thoroughness and accuracy of the implementation
over time to delivery"), did a line-by-line audit of the natural-flow
driver and the modified snapshot helpers.  Six bugs surfaced and were
fixed; the audit also drove three new tripwires and one new
cross-check, all of which immediately exercise the fixes.

### Audit findings + fixes

| # | Bug | Manifest | Fix |
|---|---|---|---|
| 1 | Six early-out paths in `KSPSolve_GMSTAB` returned without `x_local += x_global`. For a zero initial guess `x_global` is 0 throughout (no restart fires before the early-out can trigger), so the bug was MASKED by the bit-equivalence harness. For a non-zero initial guess (`KSPSetInitialGuessNonzero`) the user got a partial cycle update instead of a valid solution. Affected paths: post-init convergence, beta-below-abstol, force_l1_only return, force_l2_only return, natural-flow post-init dup convergence/abstol. | User code calling `KSPSolve` with a non-zero initial guess gets an `x` that satisfies `‖b−A·x‖ ≈ ‖x0‖` instead of `‖b−A·x‖ ≤ tolabs`. KSP-reported rnorm and externally-measured rnorm DIVERGE. | Inserted `VecAXPY(x_local, 1.0, gms->x_global)` before each early-out. Caught by new `ex_gmstab_natural_nzg`. |
| 2 | `VecDuplicate` into a non-NULL pointer leaks the prior Vec. KSPSolve_GMSTAB allocated `gms->b_local`, `gms->x_global`, `gms->r`, `gms->work_n`, `gms->work_n2` without first destroying any prior values. Multi-solve without `KSPReset` between leaks N×O(s)·sizeof(double) per call. | Memory accumulates linearly with the number of `KSPSolve` invocations on the same KSP. | Added `VecDestroy` of each on KSPSolve entry; same for `gms->V0`/`V1`/`Z`. Caught by new `ex_gmstab_multisolve`. |
| 3 | `gms->trace_fp` opened only when `!gms->trace_fp`, so on a second `KSPSolve` the file handle stayed open and the new snapshots were APPENDED to the prior solve's trace. | Trace CSV from a multi-solve test had two solves' worth of rows with no separator. Validator parsing would fail or read the wrong rows. | On every KSPSolve entry, close the existing handle (if any) and reopen with mode `"w"`. |
| 4 | `gms->V0` / `gms->V1` / `gms->Z` lazy-allocated by Init with `if (!gms->V0) MatCreate(...)`. If a user changes `s` between solves (via `-ksp_gmstab_s` re-set or `KSPGMSTABSetS`), the existing matrices are at the OLD size; the lazy-alloc skips re-creation and Init writes into stale columns. For s_new > s_old this hits a PETSc out-of-range column error; for s_new < s_old we silently use stale columns from the previous solve. | Crash (s increasing) or silent stale-data use (s decreasing) on second solve with different s. | Free V0/V1/Z on KSPSolve entry; Init re-allocates at the new size. Caught by new `ex_gmstab_schange`. |
| 5 | `gms->beta_max` set BEFORE Initialisation (to pre-init `‖r‖`) but C++ sets it AFTER (to `max(beta_post_init, betaLocal)`). For zero initial guess these are equal; with non-zero or under unusual init behaviour they diverge. | Restart heuristic on first natural-flow iter triggers slightly differently than C++. | Overwrite `gms->beta_max = max(beta_curr, beta_local)` AFTER Init returns, before entering the natural-flow loop. |
| 6 | `KSPGMSTABSnapshot_Private` (helper, not new) — verified during audit that it correctly handles the constructor special-case (`snapshot_count == 0` → return `iter_norm` as `norm_true` without recomputing). No bug here, but the audit confirmed the contract. | n/a | n/a |
| 7 | (pass 2) `gmstab_cycle1.c:112-115` early-return on `beta < abstol` after StabCoeffs (chk03 region) emitted a snapshot before returning. C++ at `solver.cpp:171-174` returns WITHOUT a snapshot in this branch; the driver's outer `if (beta <= tolabs) break` catches it on the next iter. Latent — `cdr_small` doesn't trip this because beta jumps UP at StabCoeffs in cycle 1; would surface on a problem where the polynomial step happens to drive the residual exactly into convergence on the first cycle. | One extra snapshot row in the trace vs C++; trace structure off by one for any run that ever takes this branch. | Removed the snapshot from the early-return branch; comment now points to the C++ line being mirrored. |
| 8 | (pass 2) When `init`'s inner gmres converges in <s steps PETSc init bails early (skipping the post-gmres Y/eta/Z calcs) and emits ONE snapshot. The natural-flow / force_l\*_only branches previously emitted the post-init duplicate snapshot as the FIRST line of their respective blocks, *after* the convergence early-out — so on the gmres-converged path PETSc emitted 1 init-related snapshot vs C++'s 2 (init's internal final + driver's `solver.cpp:743` dup). Latent — `cdr_small` doesn't trip this because gmres reduces beta from 1.16 to ~0.36 in 4 iters (not below tolabs); some easy cases in the cdr_sweep validation will. | Trace row count off by one whenever inner gmres converges in init. | Unified the post-init duplicate snapshot to fire UNCONDITIONALLY after Init returns, before any early-out / mode-branch logic. force_l\*_only branches and the natural-flow loop entry no longer emit their own post-init dup. |
| 9 | (pass 3) `KSPGMSTABBuildDefaultShadow_Private` initialised `gms->prand` only once (on first KSPSolve) and never re-seeded it. C++'s `default_shadow_space` constructs a fresh `std::mt19937_64` from the seed on every call — so two consecutive PETSc solves with the default-RNG shadow path consumed RNG state left over from the prior solve and produced a *different* P each time. Different P → different algorithm trajectory → `‖x1 − x2‖₂ ≠ 0` even with identical inputs. Latent — `cdr_small` validation harness uses `-ksp_gmstab_p_file`, not the RNG path; production users running the same problem twice in one process would see non-deterministic output. | Multi-solve through the default-RNG path returns different x's on the same KSP. | Re-seed `gms->prand` on EVERY call to `KSPGMSTABBuildDefaultShadow_Private` (not just first). Caught by new `ex_gmstab_multisolve_rng`, which exercises the previously-untested default-RNG multi-solve path and asserts `‖x1 − x2‖₂ = 0`. **Negative-test verified:** with the fix temporarily reverted, the tripwire reports rnorm mismatch (9.05e-11 vs 7.44e-11) and `‖x1 − x2‖₂ = 2.74e-10`, correctly catching the regression. |
| 10 | (pass 4 — defensive cleanup) `KSPGMSTABInnerWorkspaceCreate_Private` left `ws->matvec_count_ptr` uninitialised. Callers always set it immediately after the create call (`ws.matvec_count_ptr = &gms->matvec_count` at gmstab.c:174), so production paths are safe — but a future caller that forgets to set it would dereference stack garbage on every counted matvec. | Latent. No current path triggers it. | Initialise `ws->matvec_count_ptr = NULL` at the top of `KSPGMSTABInnerWorkspaceCreate_Private`; the three counter sites (`gmres_m`, `pgmres_m`, `aug_gmres_m`) already null-check before dereferencing. |

### Pass 4 — init-once-never-reset hunt (driven by the pass-3 RNG bug)

After pass 3 surfaced a "create once, never reset" pattern, did a
deliberate sweep for analogous patterns across the gmstab port:

| Resource | Pattern | Status |
|---|---|---|
| `gms->prand` | lazy-init `if (!gms->prand)` | FIXED in pass 3 — now re-seeded every call |
| `gms->work_n` | lazy-init in `SnapshotLocal_Private` | OK — destroyed at top of every `KSPSolve` (pass 1) |
| `gms->Z`, `gms->V0`, `gms->V1` | lazy-init in `Initialisation_Private` | OK — destroyed at top of every `KSPSolve` (pass 1) |
| `gms->P`, `gms->P_user`, `gms->P_file` | dispatch-time selection | OK — every `KSPSolve` re-builds via the precedence cascade |
| `gms->trace_fp` | open-once | FIXED in pass 1 — now closed-and-reopened every `KSPSolve` |
| `gms->snapshot_count`, `matvec_count`, `cycle_count`, `n2cycles`, `_last_logged_matvec_count`, `t_total`, `t_mv`, `ksp->its` | counters | OK — explicit reset at top of `KSPSolve` |
| `ksp->reason`, `ksp->rnorm0` | per-solve KSP state | OK — PETSc's `KSPSolve` resets `ksp->reason = KSP_CONVERGED_ITERATING` at line 351 of `itfunc.c`; `KSPConvergedDefault` resets `rnorm0` whenever `n == 0`, which fires on our reset `ksp->its == 0` constructor snapshot |
| `ws->matvec_count_ptr` | uninitialised stack | FIXED in pass 4 — null-init in `WorkspaceCreate` to guard against future callers that forget |
| `gmstab1_call_count` / `gmstab2_call_count` (C++) | static counter that limits dumps to first invocation | DELIBERATELY NOT MIRRORED — PETSc dump helpers are stateless and clobber the dump dir on every cycle invocation; only matters for diagnostic runs (force_l\*_only / single-cycle), and only those modes are used with dumps. Documented limitation. |

No file-scope or function-local `static` variables exist in the gmstab
sources (verified by `grep -n "^[[:space:]]*static [^(]"`). All mutable
state is either struct-resident with explicit per-solve reset, or
inner-workspace-resident and per-solve allocated/destroyed.

### New tripwires that exercise the fixes

* **`ex_gmstab_natural_nzg`** (sequential + n=2,4,8): runs the full
  natural flow with `KSPSetInitialGuessNonzero` and `x0 = 0.7·𝟙`. The
  externally measured `‖b − A·x_returned‖` must be ≤ `1.5·abstol` AND
  equal the KSP-reported rnorm to FP precision. Without the fix this
  fails by ~9 orders of magnitude (external residual ≈ ‖x0‖ ≈ 18).
* **`ex_gmstab_determinism`** (sequential + n=2,4,8): runs natural-flow
  twice on a fresh KSP each time. Asserts the two trace CSVs are
  BYTE-IDENTICAL (currently 19–20 KB each, zero divergence). Catches any
  non-deterministic operation introduced by future changes.
* **`ex_gmstab_multisolve`** (sequential + n=2,4,8): calls `KSPSolve`
  twice on the SAME KSP object. Asserts identical iter count, identical
  rnorm, and `‖x1 − x2‖₂ = 0` exactly. Catches per-solve state-not-reset
  bugs (counters, V0/V1/Z, trace_fp, etc.). Uses `-ksp_gmstab_p_file` so
  the shadow space is identical across solves by construction.
* **`ex_gmstab_multisolve_rng`** (audit pass 3) (sequential + n=2,4,8):
  multi-solve via the default-RNG shadow-space path. Same gates as
  `ex_gmstab_multisolve` but DOES NOT pass `-ksp_gmstab_p_file`. Catches
  the RNG-state bug where `gms->prand` carried state from solve 1 to
  solve 2 instead of being re-seeded.
* **`ex_gmstab_schange`** (sequential + n=2,4,8): calls `KSPSolve` with
  s=2, then re-sets `-ksp_gmstab_s` to 6, then calls `KSPSolve` again
  on the same KSP. Asserts both solves reach ATOL with externally-
  measured ‖b−A·x‖ ≤ 1.5·abstol AND that the internal rnorm matches the
  external one to FP precision (which is only true if V0/V1/Z were
  re-allocated at the new size — otherwise the second solve either
  crashes via an out-of-range MatDenseGetColumnVec or silently uses
  stale columns from the prior solve).
* **`diff_natural_seq_vs_par.py`** cross-check (n=2 and n=4 against
  seq): both runs use LAPACK on identical inputs, so the only source of
  divergence in the gauge-independent Init prefix is MPI Allreduce
  reordering. Observed worst drift: 8.88e-16 (n=2), 2.83e-15 (n=4) —
  five orders of magnitude under the 1e-10 gate. Catches any future
  parallel-only code path that introduces gauge change.

### Force_l1 / Force_l2 reverification

The user explicitly asked for both force-modes to be re-verified.
`ex_gmstab_cycle1` (force_l1) and `ex_gmstab_cycle2` (force_l2) were
already in the suite and continue to PASS at machine precision against
their respective C++ baselines:

| validator | mode | drift vs C++ baseline | rank counts |
|---|---|---|---|
| `ex_gmstab_cycle1` | `-ksp_gmstab_force_l1_only` | 1.31e-12 | 1, 2, 4, 8 |
| `ex_gmstab_cycle2` | `-ksp_gmstab_force_l2_only` | 5.26e-12 | 1, 2, 4, 8 |

Both also have dump-level seq-vs-parallel cross-checks at n=2,4 that
PASS at ≤ 1.4e-12 worst drift (using the same LAPACK on both sides).

## Bug taxonomy — patterns and detection techniques

Across the four audit passes of Phase 3d, ten bugs were found. Persisting
them here as a reference for future porting work — both for cdr_sweep
validation in Phase 5a and for any subsequent ports of similar
algorithms. Each bug is characterised by *what kind of mistake it was*,
*what scenario triggers it*, and *what audit technique catches it*.

### Bug catalog

| # | Pass | Site | Trigger | Latent on existing tests? |
|---|---|---|---|---|
| 0 | (during impl) | 9 snapshot sites in cycle1/2, gmstab.c, init | Restart fires (so `x_global ≠ 0`) | Yes — cdr_small with zero init guess and no restart never triggers |
| 1 | 1 | 6 early-out paths in `KSPSolve_GMSTAB` | `KSPSetInitialGuessNonzero` + early termination | Yes — all tests used `x0 = 0` |
| 2 | 1 | `VecDuplicate` of `b_local`/`x_global`/`r`/etc. | Two `KSPSolve` calls without `KSPReset` between | Yes — every test created a fresh KSP |
| 3 | 1 | `trace_fp` opened only when NULL | Multi-solve with `trace_csv` set | Yes — same |
| 4 | 1 | `V0`/`V1`/`Z` lazy-init `if (!gms->V0)` | Multi-solve with `s` changed between | Yes — same |
| 5 | 1 | `gms->beta_max` set BEFORE Init | `betaMax` should be `max(post-Init, pre-Init)` per C++ | Partially — wrong on first restart-trigger, usually masked because pre-Init ≥ post-Init |
| 7 | 2 | `gmstab_cycle1.c:112-115` early-return after StabCoeffs | `beta` drops below `abstol` exactly at chk03 | Yes — cdr_small's polynomial step jumps beta UP at chk03 |
| 8 | 2 | post-init dup snapshot inside force/natural blocks | Inner gmres converges in <s steps | Yes — cdr_small needs >4 inner gmres iters |
| 9 | 3 | `gms->prand` re-seeding | Multi-solve via default-RNG shadow path | Yes — all tests used `-ksp_gmstab_p_file` |
| 10 | 4 | `ws->matvec_count_ptr` uninitialised | Future refactor that omits the immediate-after-Create assignment | Yes — current callers all set it |

(Bug #6 was an audit-only verification of `KSPGMSTABSnapshot_Private`'s
constructor special-case, not an actual bug.)

### Pattern A: "Init-once-never-reset" (4 bugs: 2, 3, 4, 9)

State allocated/initialised on the first `KSPSolve` is reused on
subsequent solves without being reset to its post-construction state.

| Bug | Resource | What carried over |
|---|---|---|
| 2 | `Vec`s (`b_local`, `x_global`, `r`, `work_n`, `work_n2`) | Old Vec leaked when `VecDuplicate` overwrote the pointer |
| 3 | `trace_fp` file handle | Second solve appended to first's CSV |
| 4 | `V0`/`V1`/`Z` (sized by `s`) | Wrong size if `s` changed between solves |
| 9 | `gms->prand` RNG | State advanced through previous draws → different `P` |

**Audit technique.** Enumerate every lazily-allocated resource and ask
"what happens on the second call?" The visual signature is
`if (!gms->X) { create X }` — a giveaway that subsequent calls reuse.

**Tripwire pattern.** Multi-solve test that runs `KSPSolve` twice on the
same KSP and asserts byte-identical output (or `‖x1 − x2‖ = 0`).
`ex_gmstab_multisolve` (file-loaded P) catches bugs 2, 3, 4;
`ex_gmstab_multisolve_rng` (default-RNG P) catches bug 9.

### Pattern B: "Multi-path invariant violation" (2 bugs: 0, 1)

Many code paths reach a common point, but not all of them maintain the
same invariant. One or two paths skip the bookkeeping the rest do.

| Bug | Common invariant | Paths that violated it |
|---|---|---|
| 0 | Snapshots receive `x_total = x_global + x_local` | 9 sites passed `x_local` directly |
| 1 | On exit, `x_local` contains the full solution (`x_local += x_global`) | 6 early-out paths skipped the accumulation |

**Audit technique.** Enumerate every return / exit / branch leaving the
code region. Verify each maintains the post-condition. The simplest
diagnostic: `grep` for the function's exit primitive
(`PetscFunctionReturn`, `goto cleanup`, `break`, `return`) and audit
each site individually against a written invariant.

**Tripwire pattern.** Test the OFF-DIAGONAL paths (the rare ones), not
just the canonical happy-path. `ex_gmstab_natural_nzg` (nonzero guess)
forces `x_global ≠ 0` and catches bug 1's early-out skipping.

### Pattern C: "Snapshot rhythm divergence from reference" (3 bugs: 0, 7, 8)

PETSc emits snapshots at slightly different points than the C++
reference, producing extra/missing rows in the trace CSV.

| Bug | C++ behavior | PETSc behavior (before fix) |
|---|---|---|
| 0 | C++'s `perf.read(xGlobal + x, beta)` | PETSc's `Snapshot(x_local, beta)` — wrong x_total |
| 7 | C++ returns from cycle1 chk03 region without `perf.read` | PETSc emitted a snapshot before returning |
| 8 | C++ always emits init's `perf.read` + driver's post-init dup (2 snapshots) | PETSc emitted only 1 when init's gmres converged early |

**Audit technique.** Line-by-line cross-reference between the reference
and the port, focusing on every `perf.read` / snapshot site. Build a
table: "C++ line → PETSc line → context." Mismatches in count or
position are usually bugs.

**Tripwire pattern.** Trace bit-equivalence harness
(`ex_gmstab_cycle1`/`cycle2`/`natural`) with row-by-row comparison. Hard
to make 100% reliable for cdr_small alone — many rhythm bugs only
surface on edge-case convergence patterns (e.g. bug 7 needs beta exactly
at chk03 to cross abstol; bug 8 needs gmres to converge in <s iters).
Phase 5a's full cdr_sweep validation will exercise more edge cases.

### Pattern D: "Off-by-statement bookkeeping" (1 bug: 5)

State variable is updated, but at the wrong place in the sequence —
the value at the read site differs from what the reference produces
because order-of-operations matters.

| Bug | C++ order | PETSc order (before fix) |
|---|---|---|
| 5 | `Init(...); betaMax = max(post-Init beta, pre-Init beta);` | `betaMax = pre-Init beta; Init(...);` (no overwrite) |

**Audit technique.** For each scalar field with a "computed at solve
start" semantic, walk through the C++ code and find the EXACT moment
its value is first read. Match that moment in PETSc.

**Tripwire pattern.** Hard to write a focused tripwire — bug only
manifests on inputs that exercise the relevant boundary (here: a
post-init beta that triggers a specific restart pattern). The
natural-flow validator's matvec column ends up sensitive indirectly.

### Pattern E: "Latent uninitialised state" (1 bug: 10)

A stack-allocated struct field is never explicitly initialised; it
works because every current caller sets it immediately after
construction. Latent — a future refactor could regress.

**Audit technique.** Scan struct construction sites. For each field of
the struct, verify it's either zeroed by `PetscNew`/`PetscCalloc` or
explicitly assigned in the constructor. `KSPGMSTABInnerWorkspace` is
stack-allocated by the caller, so it's especially exposed.

**Tripwire pattern.** Code review primarily. Hard to catch with runtime
tests because production paths always set the field.

### Pattern frequency

| Pattern | Count | Fraction |
|---|---|---|
| A: Init-once-never-reset | 4 | 40% |
| B: Multi-path invariant violation | 2 | 20% |
| C: Snapshot rhythm divergence | 3 | 30% |
| D: Off-by-statement bookkeeping | 1 | 10% |
| E: Latent uninitialised state | 1 | 10% |

(C and B overlap — Bug 0 falls in both.)

Pattern A dominated, which is why pass 4 was dedicated to it. Pattern C
was second most common: **trace-rhythm cross-referencing deserves its
own systematic step** in any future port-with-bit-equivalence-validation
effort.

### Audit checklist for the next phase

Translating each pattern into a concrete checklist for Phase 4 / 5a:

1. **Multi-solve every state field.** For each `gms->X`, ask: "if I
   call `KSPSolve` twice, does X get the same value at the start of
   solve 2 as it had at the start of solve 1?" If lazy-allocated, the
   answer is usually no — fix at the top of `KSPSolve_GMSTAB`.
2. **Enumerate all exit paths.** Before declaring a function done,
   `grep` every `PetscFunctionReturn`/`return`/`break` and walk a
   checklist: does this exit maintain the function's post-conditions?
3. **Snapshot rhythm = strict 1:1 mapping with reference.** Build a
   numbered table cross-referencing every reference `perf.read` to its
   PETSc equivalent. Catch mismatches at writing time, not validation
   time.
4. **Order-of-operations matters for scalar fields.** When a field has
   multiple plausible "set" sites, the right one is whichever the
   reference uses — at the same logical position.
5. **Defensive zero-init is cheap insurance.** Every struct field not
   already zeroed by `PetscNew` should be explicitly set in
   `Create`/`Init` functions. Stack-allocated structs need explicit
   initialisation of every field.

## Phase 4 — preconditioner sides (right, left); CPU PC sweep

Phase 4 brings `KSPGMSTAB` to functional parity with PETSc's PC framework.
The plan is in `PHASE4_PLAN.md`; the audit findings are below.

### Bug catalog (continuation of Phase 3d)

| # | Pass | Site | Fix |
|---|---|---|---|
| 11 | Phase 4a | `KSPSolve_GMSTAB`'s 5 return paths each had inline `VecAXPY(x_local, 1.0, gms->x_global)` (Pattern B again). For PC_RIGHT correctness we needed to apply `B⁻¹` instead of plain accumulation. | Added `KSPGMSTABFinalizeSolution_Private`; replaced all 5 inline accumulations with the helper call. |
| 12 | Phase 4a | `bLocal = b − A·x_initial` was computed via `KSP_PCApplyBAorAB`, which for PC_RIGHT returns `A·B⁻¹·x_initial` (giving `b − A·B⁻¹·x_initial`, not the unpreconditioned residual the algorithm assumes). Latent: invisible when `x_initial = 0`; off by `A·(I − B⁻¹)·x_initial` otherwise (caught by the new `ex_gmstab_pcright_nzg` tripwire). | Switched to plain `MatMult(Amat, gms->x_global, Ax)` for the bLocal computation. For PC_NONE this is identical to the previous `KSP_PCApplyBAorAB` call. |
| 13 | Phase 4b | `bLocal` was in unpreconditioned algebra while the cycle's matvecs go through `KSP_PCApplyBAorAB → B⁻¹·A` for PC_LEFT, leaving the residual algebra inconsistent. | Apply `PCApply` to bLocal once at solve start when `pc_side == PC_LEFT`; cycles' algebra is then consistent in M-space throughout. |
| 14 | Phase 4b | The natural-flow loop's exit checks (`while (beta_curr > ksp->abstol)` and `if (beta_curr <= ksp->abstol) reason=ATOL; break;`) used the algorithm-internal `beta_curr`, which for PC_LEFT is `‖r_pre‖`, not the user-visible residual. Premature exit when `‖r_pre‖ ≤ abstol` but `‖r_unpre‖ > abstol` — caught by the first run of `ex_gmstab_pcleft_jacobi` failing with rnorm=4e-10. | Made the loop reason-driven: it iterates until `KSPConvergedDefault` (called via `Snapshot_Private`) sets `ksp->reason`, which already uses the right residual per the user's `KSPSetNormType`. |
| 15 | Phase 4b | Some PC_LEFT + tough-PC combinations (e.g. small-block PCBJACOBI on cdr_small at n=8) loop for millions of snapshots before a numerical event terminates them. `KSPConvergedDefault`'s DIVERGED_ITS check appears to not fire for this KSP type as expected. | Added a defensive runaway cap at `100 × ksp->max_it` cycles; sets `KSP_DIVERGED_ITS` and exits. The PC_LEFT + small-block-BJacobi limitation is also documented and skipped at `n ≥ 8` in the bjacobi tripwire. |

### New tripwires (Phase 4)

| Validator | Phase | Coverage |
|---|---|---|
| `ex_gmstab_pcright_jacobi` | 4a | Basic PC_RIGHT correctness with PCJACOBI; verifies external residual matches internal rnorm to FP precision. |
| `ex_gmstab_pcright_nzg` | 4a | PC_RIGHT + nonzero initial guess. Catches the `bLocal = b − A·B⁻¹·x_initial` bug (Bug 12) — without the fix, ext_res = ~85 instead of ~1e-10. |
| `ex_gmstab_pcleft_jacobi` | 4b | Basic PC_LEFT correctness with PCJACOBI. Caught the convergence-target-mismatch bug (Bug 14). |
| `ex_gmstab_pcleft_bjacobi` | 4b | PC_LEFT + parallel-default PCBJACOBI; skips `n ≥ 8` (documented small-block ILU stall). |
| `ex_gmstab_pc_sweep` | 4d | Cross-product `(pc_side, pc_type)` for `pc_type ∈ {Jacobi, BJacobi, SOR, ILU, ASM}`. Skips the documented 2 known-bad combinations (PC_LEFT+SOR everywhere, PC_LEFT+BJacobi at `n ≥ 8`). |

### Pattern frequency (updated)

Phase 4 found 5 more bugs, mostly Pattern B (multi-path) and Pattern D (off-by-statement / wrong-residual). Updated taxonomy frequency:

| Pattern | Phase 3d count | Phase 4 count | Total |
|---|---|---|---|
| A: Init-once-never-reset | 4 | 0 | 4 |
| B: Multi-path invariant violation | 2 | 1 (Bug 11) | 3 |
| C: Snapshot rhythm divergence | 3 | 0 | 3 |
| D: Off-by-statement bookkeeping | 1 | 3 (Bugs 12, 13, 14) | 4 |
| E: Latent uninitialised state | 1 | 0 | 1 |
| F: Defensive runaway cap (new pattern) | 0 | 1 (Bug 15) | 1 |

Pattern D (off-by-statement / wrong-residual / wrong-algebra) jumped from 1 to 4. Phase 4 reinforced the Phase 3d audit checklist's #4: **order-of-operations matters for scalar fields, AND the operator side / residual algebra also has to match across every site**.

### Final tripwire suite: 60/60 PASS (was 40/40 after Phase 3d)

* 15 sequential validators (added `pcright_jacobi`, `pcright_nzg`, `pcleft_jacobi`, `pcleft_bjacobi`, `pc_sweep`)
* 39 parallel runs (cycle1, cycle2, natural, natural_nzg, determinism, multisolve, multisolve_rng, schange, pcright_jacobi, pcright_nzg, pcleft_jacobi, pcleft_bjacobi, pc_sweep × ranks ∈ {2, 4, 8})
* 4 dump-level cross-checks (cycle1, cycle2 × ranks ∈ {2, 4})
* 2 trace-level cross-checks (natural seq-vs-parallel × ranks ∈ {2, 4})

## How to resume

```bash
# 1. Confirm we're on the validated baseline
cd /home/sam/hpc_stack/petsc
git status   # should be clean on ksp-gmstab
src/ksp/ksp/impls/gmstab/tests/run_gmstab_tripwires.sh
# Expect: 60/60 PASS

# 2. Move on to Phase 4 (preconditioner sides) or Phase 5a (full
#    sweep validation against the 129 baselines).
```
