# GMSTAB — User Manual

**KSP type:** `KSPGMSTAB`
**Status:** CPU production-ready (Phase 4); GPU and full bit-equivalence sweep pending
**Reference algorithm:** GMstab (IDR(s)-family Krylov solver with flying restart),
ported from Sleijpen/Sosonkina/Saad's MATLAB and C++ reference implementations.

This manual covers the production-facing usage of the PETSc GMSTAB port. For the
algorithm's design rationale and known limitations, see `PHASE3_STATUS.md`. For the
PC-side × norm-type interaction discussion, see `PC_LEFT_NORM_DISCUSSION.md`.

---

## 1. Quick start

```c
KSP ksp;
KSPCreate(PETSC_COMM_WORLD, &ksp);
KSPSetOperators(ksp, A, A);
KSPSetType(ksp, KSPGMSTAB);

/* Recommended for non-symmetric / advection-dominated problems: */
KSPSetPCSide(ksp, PC_RIGHT);
KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);

PC pc;
KSPGetPC(ksp, &pc);
PCSetType(pc, PCBJACOBI);     /* or PCILU / PCASM / PCJACOBI / PCSOR */

KSPSetTolerances(ksp, 0.0, 1e-10, PETSC_DEFAULT, 5000);
KSPSolve(ksp, b, x);
```

Command-line option counterparts:

```
-ksp_type gmstab
-ksp_pc_side right
-ksp_norm_type unpreconditioned
-pc_type bjacobi
-ksp_atol 1e-10
-ksp_max_it 5000
```

GMSTAB-specific options:

```
-ksp_gmstab_s 4              # IDR shadow-space dimension (default: 4)
-ksp_gmstab_p_file P.bin     # explicit shadow space (column-major float64)
-ksp_gmstab_rng_seed 5489    # RNG seed when generating shadow space (default: 5489)
-ksp_gmstab_force_l1_only    # validation-only: run one Cycle1 then exit
-ksp_gmstab_force_l2_only    # validation-only: run one Cycle2 then exit
-ksp_gmstab_trace_csv path   # debug: dump per-snapshot residual trace to CSV
```

---

## 2. Choosing preconditioner side and norm type

This is the one decision a GMSTAB user really needs to get right. The algorithm has
two natural pairings, and a third "cross-pairing" that costs extra MatMults per
iteration:

| Configuration | Cost per cycle | Convergence behavior | When to use |
|---|---|---|---|
| `PC_RIGHT` + `KSP_NORM_UNPRECONDITIONED` | 0 extra MatMults | Native | **Default recommendation.** The convergence check uses `||b − A·x||`, the algorithm tracks `||b − A·x||`. No mismatch. |
| `PC_LEFT` + `KSP_NORM_PRECONDITIONED` | 0 extra MatMults | Native | Use if you don't care about the user-visible residual norm and want maximum speed. The algorithm tracks `||B⁻¹·(b − A·x)||`. |
| `PC_RIGHT` + `KSP_NORM_PRECONDITIONED` | +1 PCApply per snapshot | Native | Rarely useful; PC_RIGHT's natural norm is the unpreconditioned one. |
| `PC_LEFT` + `KSP_NORM_UNPRECONDITIONED` | **+4–6 MatMults per cycle** | Can stall under weak PCs | **Avoid if possible.** See §3 below for details and workarounds. |
| `PC_NONE` + (any) | 0 extra MatMults | Native | Equivalent under PC_NONE. |

### Recommendation

- **If you need to compare GMSTAB's reported residual against an outer-loop residual
  (Newton solver, time-stepping, etc.):** use `PC_RIGHT` + `KSP_NORM_UNPRECONDITIONED`.
  This is the default recommendation and the most common configuration.

- **If you don't care which residual norm is reported and want maximum speed:** use
  `PC_LEFT` + `KSP_NORM_PRECONDITIONED`. Performance-equivalent to PC_RIGHT but with a
  different residual reported.

- **If you're debugging or running validation against the C++ baseline:** use
  `PC_LEFT` + `KSP_NORM_PRECONDITIONED`. The C++ reference operates in this algebra
  natively, so this gives bit-equivalent traces.

---

## 3. Warning — `PC_LEFT` + `KSP_NORM_UNPRECONDITIONED` cost and stall

**This combination is supported but expensive and can stall on weak preconditioners.**

### Why it costs more

Under PC_LEFT, the GMstab cycle works in *preconditioned* algebra: the algorithm
internally tracks `||r_pre|| = ||B⁻¹·(b − A·x)||`. The user asked for the
*unpreconditioned* norm `||b − A·x||`, which the algorithm doesn't have natively.

To honor `KSP_NORM_UNPRECONDITIONED`, `KSPGMSTABSnapshot_Private` performs an extra
`MatMult(A, x_user, Ax)` to recompute `||b − A·x_user||` from scratch on every
snapshot. The IDR(s) cycle structure fires 4–6 snapshots per outer iteration (after
the τ-step, polynomial step, pGMRESm inner solve, etc. — see
`PC_LEFT_NORM_DISCUSSION.md` Q14 for the full cadence).

So the cost overhead is **4–6 extra MatMults per outer iteration**, on top of the
algorithm's algorithmic MatMults. If MatMult dominates your wall time (typical for
large sparse systems), this is roughly **4–6× wall-time overhead** vs. the natural
pairing.

### Why it can stall

The cycle's internal short-circuit at `gmstab.c:287/308/332` and `cycle1.c:118`,
`cycle2.c:158/476` exits when `gms->beta = ||r_pre|| ≤ ksp->abstol`. But the user's
convergence check (via `KSPConvergedDefault` inside `Snapshot_Private`) tests
`||r_unprec|| ≤ ksp->abstol`. These two scalars can differ by the conditioning ratio
of `B⁻¹` against the residual:

```
||r_pre|| / ||r_unprec|| ≈ ||B⁻¹|| in the residual's principal directions
```

For Jacobi (`B = diag(A)`) on advection-dominated problems, that ratio runs ~10×.
The cycle exits when `||r_pre|| ≤ 1e-10`, but `||r_unprec||` can stall at ~1e-9 — and
because the cycle keeps short-circuiting, the algorithm never refines further.
Concretely, you'll see `KSP_DIVERGED_ITS` with `rnorm` plateauing just above your
`atol`.

This is a **port artifact**, not an algorithmic bug — the C++/MATLAB references use a
single `tolabs` against `||r_pre||` because they have no separate norm-type
abstraction. PETSc's `KSPSetNormType` API exposes the choice and our port hardwires
the inline check to the algorithm-native scalar.

A planned follow-up (`PHASE4_FOLLOWUP_PLAN.md`) will remove the inline check and
delegate exit decisions to `ksp->reason` (which honors normtype). Until then:

### Workarounds

- **Switch to the natural pairing.** `PC_LEFT` + `KSP_NORM_PRECONDITIONED` will
  converge cleanly with no stall and no extra MatMults. If you need to compare to an
  outer residual, do the MatMult yourself once after `KSPSolve` returns.

- **Switch the side.** `PC_RIGHT` + `KSP_NORM_UNPRECONDITIONED` gives you the unprec
  residual natively, no stall, no extra MatMults. Iteration count is comparable to
  PC_LEFT for most preconditioners.

- **Accept a looser absolute tolerance.** If you set `atol = 1e-7` instead of `1e-10`,
  the algorithm-native `||r_pre||` and the user-visible `||r_unprec||` both drop below
  `atol` before the stall regime, and convergence is reported normally. This is the
  approach the validation tripwires take (see `tests/ex_gmstab_pcleft_jacobi.c`).

- **Bump `max_it` significantly.** Under PC_LEFT + UNPRECONDITIONED + Jacobi on
  cdr_small, full convergence to `1e-10` takes ~48,000 iterations. If wall time isn't
  the bottleneck, you can let the algorithm grind through.

---

## 4. Selecting `s` (shadow-space dimension)

`s` controls the dimension of the IDR shadow space. Larger `s` typically means:
- Faster convergence in iteration count
- More memory (algorithm stores `O(s)` extra Krylov vectors)
- More work per cycle (`O(s²)` BGS, `O(s³)` LAPACK ops)

Defaults to `s = 4`, which is the value used in the C++ reference and matches typical
literature recommendations. Use `-ksp_gmstab_s` to override:

```
-ksp_gmstab_s 8     # better convergence, more memory
-ksp_gmstab_s 2     # less memory, slower convergence
```

For most problems on cdr_small or comparably-sized systems, `s = 4` is the right
default. Larger `s` (8 or 16) can help on very stiff problems where the residual has
many slow-decay directions.

---

## 5. Initial guess

GMSTAB honors `KSPSetInitialGuessNonzero(ksp, PETSC_TRUE)`. The non-zero initial guess
is captured at solve start (`gmstab.c:188-190`) and the algorithm operates on
`x_local = x − x_initial`, then unwraps at exit (`KSPGMSTABFinalizeSolution_Private`).

Under PC_RIGHT, the unwrap also applies `B⁻¹` to the cycle's accumulated update —
this is the "Phase 4a" correctness fix; without it, the returned `x` would not satisfy
`||b − A·x|| ≈ rnorm`. Verified by the `ex_gmstab_pcright_nzg` tripwire.

---

## 6. Multi-solve and parameter changes between solves

Calling `KSPSolve(ksp, b, x)` twice on the same `KSP` is supported and produces:
- **Bit-identical results** if the inputs (`A`, `b`, initial guess) are identical
- **Independent valid solutions** if the inputs differ

Internal state (`x_initial_guess`, RNG state, BGS workspace) is correctly
reset/reallocated per-solve. Verified by `ex_gmstab_multisolve`,
`ex_gmstab_multisolve_rng`, `ex_gmstab_pc_multisolve`.

You can change `s` between solves via `-ksp_gmstab_s` or programmatically; the
algorithm will reallocate `V0`, `V1`, `Z` to the new size on the next solve. Verified
by `ex_gmstab_schange`.

You can also change `PCSide` between solves on the same `KSP` (e.g., `PC_RIGHT` first
solve, then `PC_LEFT` second). The conditional `x_initial_guess` allocation handles
this correctly. Verified by `ex_gmstab_pc_multisolve` Scenario 3.

---

## 7. Parallel execution

GMSTAB supports MPI parallelism with the standard PETSc data distribution. All
linear-algebra ops use the underlying `Mat`/`Vec` parallel layout; the algorithm has
no rank-aware logic (no rank-0 shortcuts, no special boundary handling).

Tested rank counts: 1, 2, 4, 8 (see `tests/run_gmstab_tripwires.sh`).

Sequential and parallel runs produce traces that agree to FP precision for the
gauge-independent init prefix (verified to ~5e-13 by
`tests/diff_seq_vs_parallel.py`). Post-init iterations may diverge by O(1e-12) due to
MPI reduction-order FP drift in the IDR(s) BGS step — this is expected and bounded;
correctness is preserved (the resulting `x` satisfies `||b − A·x|| ≤ rtol·||b||` at
all rank counts).

---

## 8. Known limitations

### `PC_LEFT` + `KSP_NORM_UNPRECONDITIONED` + weak PC

Documented in §3 above. Stall + 4–6 extra MatMults/cycle. Workaround: use the natural
pairing or relax atol. Permanent fix planned (`PHASE4_FOLLOWUP_PLAN.md`).

### `PC_LEFT` + `PCSOR` (default ω)

Default `ω = 1.0` is unstable for non-symmetric strongly-advection-dominated problems
under PC_LEFT. Produces NaN. Workaround: use `PC_RIGHT` (works) or tune `ω`.
Documented limitation; tripwire skips this combo (`ex_gmstab_pc_sweep.c:127-131`).

### `PC_LEFT` + `PCBJACOBI` at high rank counts on cdr_small

At MPI rank counts where the per-rank block becomes small (e.g., n=8 on N=729 → block
size ~91), the default sub-PC (KSP_PREONLY + ILU0) is too weak on advection-dominated
sub-blocks. `||r_pre||` can stall at ~7e-9 in the residual's high-conditioning
direction. Skipped at n≥8 in `ex_gmstab_pcleft_bjacobi.c`. Workaround: use `PC_RIGHT`
or stronger sub-PC.

### `PC_SYMMETRIC` supported with caveats (Phase 4c)

`KSPSetPCSide(ksp, PC_SYMMETRIC)` is supported with `KSPSetNormType(KSP_NORM_PRECONDITIONED)`
(natural pairing) or `KSP_NORM_NONE`. The cycle operates on `M = B_L⁻¹·A·B_R⁻¹`, with
`bLocal` preconditioned by `PCApplySymmetricLeft` once at solve start and the unwrap at
finalize/snapshot routed through `PCApplySymmetricRight` (only the right factor `B_R⁻¹`).

`KSP_NORM_UNPRECONDITIONED` with `PC_SYMMETRIC` is intentionally NOT declared — the
algorithm tracks `||r_pre||` natively, and converting to `||r_unprec||` would require
an extra `PCApplySymmetricLeft` per snapshot (similar cost to the PC_LEFT + UNPREC
case documented above). Use the natural pairing.

**PETSc-side limitation: not all PCs implement `PCApplySymmetricLeft/Right` cleanly.**

| PC type | `PC_SYMMETRIC` support |
|---|---|
| `PCJACOBI` | Works — symmetric square-root split, `B_L = B_R = sqrt(diag(A))` |
| `PCICC` | Works (SPD-only) |
| `PCCHOLESKY` | Works (SPD-only) |
| `PCBJACOBI` | Works **iff** sub-PC supports symmetric apply. Default sub-PC is `KSPPREONLY+PCILU`; PCILU's symmetric path is broken (see below). Use `-sub_pc_type jacobi` to make this work. |
| `PCSHELL` | Works (user provides `applysymmetricleft/right`) |
| `PCMAT`, `PCBDDC`, `PCNN`, `PCLMVM`, `PCTFS`, `PCPBJACOBI`, `PCVPBJACOBI` | Implement `applysymmetricleft/right` per source; not exercised by gmstab tests |
| `PCILU` | **PETSc-internal limitation:** `PCApplySymmetricLeft_ILU` calls `MatForwardSolve` which errors with "No method forwardsolve for Mat of type seqaij". The factored-matrix forwardsolve op isn't being attached for the symmetric path. Avoid until PETSc fixes this; use `PC_RIGHT` + `PCILU` instead. |
| `PCSOR`, `PCASM`, `PCGAMG`, `PCHYPRE` | Don't implement `applysymmetricleft/right`. PETSc errors at the dispatch layer. Use `PC_LEFT` or `PC_RIGHT`. |

Validated under cdr_small: `PC_SYMMETRIC + PCJACOBI` and
`PC_SYMMETRIC + PCBJACOBI(-sub_pc_type jacobi)` at n=1/2/4/8 ranks. See
`tests/ex_gmstab_pcsymmetric_sweep.c`.

### MATLAB split-precond baselines (Phase 5a forecast)

The reference MATLAB tests use `A_fun = @(v) L\(A*(R\v))` patterns with explicit
`L` and `R` factors. The natural Phase 5a mapping is `KSPSetPCSide(PC_SYMMETRIC) + PCSHELL`,
with `PCSHELL`'s `applysymmetricleft` set to `L⁻¹` and `applysymmetricright` set to `R⁻¹`.
This preserves the exact algebra of the reference and exercises the gmstab
PC_SYMMETRIC path.

### GPU preconditioners not yet validated

Phase 4e is deferred. The algorithm's `MatMult` and `PCApply` calls are
device-agnostic, but no end-to-end test confirms correctness with GPU `Mat` types.
CPU-only is the production-ready surface today.

---

## 9. Diagnostics

### Convergence trace

```
-ksp_monitor                  # rnorm per snapshot
-ksp_view                     # full solver state at convergence
-ksp_gmstab_trace_csv path    # detailed (iter, matvec_count, iter_norm, norm_true, ...) per snapshot
```

The trace CSV format matches the C++ reference and is consumed by
`tests/diff_gmstab_dumps.py` for bit-equivalence validation. Each row is one
`Snapshot_Private` fire — under `KSP_NORM_UNPRECONDITIONED` that includes a MatMult
to recompute the true residual, even if you only asked for the prec norm in the
algorithm's tracking.

### Cycle counts

`-ksp_view` includes:
- `cycle_count` — total outer iterations
- `n2cycles` — number of Cycle2 invocations (rest are Cycle1)
- `matvec_count` — total `MatMult(A, ·)` calls (algorithmic + norm-tracking)
- `snapshot_count` — total snapshots fired

For perf comparison with baseline, the relevant metric is **matvec_count** — wall
time is dominated by MatMult on most large sparse problems. See
`PHASE3_STATUS.md` for an explanation of how matvec_count relates to "iteration
count" in the IDR(s) literature (they aren't the same).

---

## 10. References

- **PHASE3_STATUS.md** — design decisions, known bugs and patterns, audit history
- **PC_LEFT_NORM_DISCUSSION.md** — design discussion behind §3 of this manual
- **PHASE4_FOLLOWUP_PLAN.md** — plan for the option (i) cleanup that fixes the §3 stall
- **PHASE4_PLAN.md** — Phase 4 sub-phase breakdown (4a, 4b, 4c, 4d, 4e)
- **tests/run_gmstab_tripwires.sh** — full validation suite (currently 64/64 passing)
- Sleijpen, G.L.G. and Sosonkina, M. (2010). *GMstab: Generalized Minimal Residual
  with Stabilization* — primary reference for the algorithm.
