# `KSPGMSTAB` — implementation plan archive

**Branch:** `ksp-gmstab`, branched off `amgx-block-aij-support` at commit `ff2d3d288ff`
(which itself sits on the AMGX bug-fix series: destructor recovery `2d93faec18c` +
block-AIJ support `ff2d3d288ff` + the local AMGX `m_fixed_view_size` patch in
`arch-cuda-opt-i32/externalpackages/amgx-2.4.0/include/matrix.h`).

**Author:** Sam Austin, in collaboration with Claude Opus 4.7 (1M context).

**Date archived:** 2026-05-02.

**Source-of-truth references:**
- MATLAB reference solver: `/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package/`
- Existing C++ port (validated to ≤1e-10 vs MATLAB): `/home/sam/hpc_stack/gmstab_cpp/`
- Bit-for-bit baselines (129 tests): `/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/`
- Algorithm specification: `gmstab_handoff_package/GMSTAB_HANDOFF.md`

---

## 0. Project goal

Implement GM(s)stab — a member of the IDR(s) family with a "flying restart"
between L=1 and L=2 stabilization cycles, internal use of modified GMRES
projected against the shadow space, and an angle-threshold (35°)
maintaining-the-convergence stabilizer — as a first-class PETSc KSP type
named `KSPGMSTAB`. Once registered, GMstab composes with every PETSc
preconditioner family (BoomerAMG, AMGX, GAMG, ASM, ILU, jacobi, PBJacobi,
hypre Euclid, ParaSails, Pilut, ...) and runs on any PETSc backend (CPU,
CUDA, HIP). This unlocks `KSPGMSTAB × PCAMGX` and similar combinations
that are currently impossible because GMstab lives outside the PETSc
ecosystem.

The end deliverable is an answer to the production question: **"On the
HDG production matrix (streamer_qhdg_4M7, N=4.7M, nnz=213M, bSize=9), is
KSPGMSTAB faster than KSPGMRES paired with the best preconditioner family
identified by the comparison harness?"**

---

## 1. Pre-flight asset inventory

| asset | location | status | role |
|---|---|---|---|
| MATLAB reference (`Solver_GMsStab.m` etc.) | `gmstab_matlab/gmstab_handoff_package` | working | gold standard |
| C++ port (~2233 LOC across 9 files) | `gmstab_cpp/` (`solver.cpp` 613 LOC, `modified_gmres.cpp` 343 LOC, `stab_coeffs.cpp` 101 LOC, `small_dense.cpp` 106 LOC, plus headers) | validated to ≤1e-10 vs MATLAB | line-by-line translation reference |
| Bit-for-bit baselines | `gmstab_matlab/gmstab_handoff_package_validation/baselines/{cdr_small, sherman5, ocean, asic_320ks, torso1, cdr_sweep_small ×124}` | each test ships `linsys.bin` + `P.bin` (deterministic shadow space) + `residuals.csv` (MATLAB) + `cpp_residuals.csv` (C++) + `summary.txt` | primary verification fixture |
| Validation summary template | `gmstab_handoff_package_validation/validation_summary.csv` | rolled-up C++-vs-MATLAB diff | format the PETSc port also emits |
| 575+ config sweep harness | `idr_petsc_ginkgo_comparison/` (resilient wrapper, post-AMGX-fix) | working | Phase 6/7 production target |

The single decisive fact: the C++ port and MATLAB reference *already agree
to ≤1e-10 absolute drift on `iterres` on every baseline*. Phase 5a's
correctness criterion is therefore well-defined and reproducible — we
have a bit-level oracle for the port.

---

## 2. Architectural translation map (Eigen → PETSc)

| C++ thing | size in GMstab | PETSc replacement | rationale |
|---|---|---|---|
| `Vec b, x, r0` (length N) | distributed | `Vec` | first-class |
| `Mat V0, V1` (N × s+1) Krylov-block | tall-skinny, distributed | `MatDense` | works with `MatMatMult(A, V, ...)` for vectorized matvec; cuSPARSE has batched SpMM under MATAIJCUSPARSE |
| `Mat W` (N × s+1) inner GMRES basis | tall-skinny, distributed | `MatDense` | same |
| `Mat Z` (s × s) | small dense | host raw `PetscScalar*` | small enough that `BLASgemm_`/`LAPACKgesv_` direct beats PETSc-Mat overhead |
| `Mat P` (N × s, shadow space) | tall-skinny, distributed | `MatDense` | same as V0/V1 |
| `Mat H, Q, R` from inner GMRES | small dense | host raw arrays + LAPACK | same |
| `LinOp A_fun` (functor) | — | `KSP_PCApplyBAorAB(ksp, x, y, work)` | routes through PC side automatically |
| `PerfMeasure` (matvec counter, runtime) | — | `KSPLogResidualHistory` + custom monitor | matches PETSc convention |
| `default_shadow_space(N, s, seed)` | — | `std::mt19937_64` (matches C++) + Gram-Schmidt via `VecMDot`/`VecAXPY` | bit-equivalent shadow space generation |
| Eigen LQ decomp (`lq()`) | s × s | `LAPACKgelqf_` (or `LAPACKgeqrf_` of transpose, with explicit Q reconstruction) | needs sign convention check vs Eigen |
| Eigen `linsolve` | s × s | `LAPACKgesv_` | direct |
| Eigen `triangularView<Lower>().solve()` | s × s | `BLAStrsv_` / `BLAStrsm_` | direct |

**Two operations to watch carefully:**

1. **`MatMatMult` with wide MatDense `V` as right operand.** PETSc's
   `MatMatMult(A, V, MAT_INITIAL_MATRIX, fill, &W)` handles `Mat A` ×
   `MatDense V` and produces a `MatDense`. The CUSPARSE path uses
   cuSPARSE's batched SpMM. **We must NOT loop column-wise** — that
   would defeat the point of using PETSc's vectorized matvec on GPU.

2. **`P^T W` (small dense from tall-skinny dot products).** This is
   `MatTransposeMatMult` or `MatTDotMat`. PETSc has both, but we want
   the result on the *host* as a small dense block for downstream
   LAPACK ops. Allocate a `PetscScalar*` host buffer and download via
   `MatDenseGetArrayRead` after the multiply.

---

## 3. The PETSc `KSPType` contract

### 3.1 File layout

```
src/ksp/ksp/impls/gmstab/
├── makefile                    # 3 LOC, copy from bcgsl
├── gmstab.c                    # ~250 LOC: KSPCreate/SetUp/Solve/Destroy/SetFromOptions/View/Reset
├── gmstab_init.c               # ~150 LOC: Initialisation (port of Initialisation.m)
├── gmstab_cycles.c             # ~400 LOC: GMstab1, GMstab2 (the two cycle bodies)
├── gmstab_modgmres.c           # ~500 LOC: pGMRESm + augGMRESm (the inner GMRES variants)
├── gmstab_stab.c               # ~120 LOC: StabCoeffs (35° polynomial stabilization)
├── gmstab_lq.c                 # ~80 LOC: small-dense host-LAPACK helpers (lq, qr, linsolve)
├── gmstabimpl.h                # ~80 LOC: private struct + macros
├── tests/                      # ex_gmstab_validate.c + unit tests
└── IMPLEMENTATION_PLAN.md      # this file
```

Closest existing cognate in PETSc: `src/ksp/ksp/impls/bcgsl/` (BiCGStab(L),
structurally similar to IDR(s)stab — uses an L-degree polynomial stabilizer).
Use it as the syntactic template.

### 3.2 `KSP_GMSTAB` private struct (in `gmstabimpl.h`)

```c
typedef struct {
  /* User-facing parameters (configured via -ksp_gmstab_*) */
  PetscInt    s;            /* shadow-space dimension, default 4 */
  PetscInt    L;            /* stabilization polynomial degree (informational; 1 or 2) */
  PetscReal   tolabs2;      /* recycling threshold; 0 = recycling disabled (Phase 8 only) */
  PetscReal   stab_angle;   /* maintaining-the-convergence threshold, default 35° */
  PetscBool   force_initial_cycle1; /* debugging knob */
  char        P_file[PETSC_MAX_PATH_LEN]; /* deterministic P override (validation) */
  PetscRandom prand;        /* for reproducible default P */

  /* Persistent solver state across iterations */
  Mat         P;            /* N × s shadow space (dense) */
  Mat         V0, V1;       /* N × s+1 Krylov blocks (dense) */
  Mat         W;            /* N × s+1 inner GMRES basis (dense) */
  Vec         r;            /* current residual */
  PetscReal   beta;         /* ||r|| (true residual norm; same as iterres for no-PC and right-PC) */
  PetscReal   beta_pc;      /* preconditioned-residual norm, tracked when pc_side != NONE */
  Vec         work_n;       /* one length-N scratch */
  Mat         work_s;       /* small s+1 × s+1 scratch (dense, host) */

  /* Recycling (Phase 8 only) */
  Mat         hU;
  PetscBool   has_recycling;

  /* Cycle selection (flying restart) */
  PetscInt    cycle_count;
  PetscReal   last_cycle_residual;

  /* Bit-equivalent validation: per-iter trace */
  PetscBool   trace_csv;
  char        trace_csv_path[PETSC_MAX_PATH_LEN];
  FILE       *trace_fp;
} KSP_GMSTAB;
```

### 3.3 Entry points

```c
PETSC_INTERN PetscErrorCode KSPCreate_GMSTAB(KSP);
static PetscErrorCode KSPSetUp_GMSTAB(KSP);
static PetscErrorCode KSPSolve_GMSTAB(KSP);
static PetscErrorCode KSPDestroy_GMSTAB(KSP);
static PetscErrorCode KSPSetFromOptions_GMSTAB(KSP, PetscOptionItems);
static PetscErrorCode KSPView_GMSTAB(KSP, PetscViewer);
static PetscErrorCode KSPReset_GMSTAB(KSP);
```

Plus two public APIs for Phase 5 validation:

```c
PetscErrorCode KSPGMSTABSetShadowSpace(KSP ksp, Mat P);     /* programmatic */
PetscErrorCode KSPGMSTABSetShadowSpaceFile(KSP ksp, const char *path); /* P.bin loader */
```

### 3.4 Registration (3 lines)

- `include/petscksp.h`: `#define KSPGMSTAB "gmstab"`
- `src/ksp/ksp/interface/itregis.c`: forward decl + `KSPRegister(KSPGMSTAB, KSPCreate_GMSTAB)`
- `src/ksp/ksp/impls/makefile`: add `gmstab` to DIRS

After registration, `-ksp_type gmstab` works everywhere PETSc is used —
from C, Python (petsc4py), the comparison harness sweep CSV, etc.

---

## 4. Resolved design questions

### 4.1 Preconditioner sides — implement all three

`KSP_PCApplyBAorAB(ksp, x, Ax, t)` (declared in `petsc/private/kspimpl.h`)
applies the operator the Krylov solver should treat as `A`, hiding `pc_side`
behind a uniform interface:

| `pc_side` | what `KSP_PCApplyBAorAB` computes | natural residual the algorithm tracks |
|---|---|---|
| `KSP_PC_LEFT` | `M⁻¹ A x` | `M⁻¹ (b − Ax)` (preconditioned residual) |
| `KSP_PC_RIGHT` | `A M⁻¹ x` | `b − Ax` (true residual) |
| `KSP_PC_SYMMETRIC` (split) | `M_L⁻¹ A M_R⁻¹ x` (only when PC supports `PCApplySymmetric`, e.g. `PCICC`, `PCSOR`) | `M_L⁻¹ (b − Ax)` |

GMstab is unusual among IDR(s) family solvers in that it **explicitly tracks
the true residual `r = b − Ax` internally** (the C++ port's `r0` is updated
as `r0 ← r0 − A_fun(dx)` where `dx` is in the original solution space).
This makes **right preconditioning the natural choice** — the inner-loop
arithmetic is unchanged; we substitute `Ax → A M⁻¹ x` for the matvec and
apply `x_actual = M⁻¹ x` once at the end.

For left and split, the algorithm needs an extra `M`-apply per residual
update to recover the true residual when `-ksp_norm_type unpreconditioned`
is in effect. Bookkeeping, but adds PC applies.

**Implementation order, deliberately staged within Phase 4:**

| sub-phase | sides supported | smoke test |
|---|---|---|
| 4a | right precond | `pc_type jacobi` + `-ksp_pc_side right` (default) on cdr_small |
| 4b | + left precond | `pc_type jacobi` + `-ksp_pc_side left` |
| 4c | + split precond | `pc_type icc` + `-ksp_pc_side symmetric` |

`KSPSetSupportedNorm` priority registrations in `KSPCreate_GMSTAB`:

```c
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT,     3));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED,   PC_RIGHT,     2));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT,      2));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED,   PC_LEFT,      3));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED,   PC_SYMMETRIC, 2));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_SYMMETRIC, 1));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE,             PC_LEFT,      1));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE,             PC_RIGHT,     1));
PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE,             PC_SYMMETRIC, 1));
```

Default registered as priority-3 for `(PC_RIGHT, NORM_UNPRECONDITIONED)`,
matching the convention the comparison harness uses.

### 4.2 True residual vs preconditioned residual for convergence check

GMstab tracks the **true residual** internally, so it can deliver either
norm essentially for free. The choice is controlled by `-ksp_norm_type`
exactly as PETSc convention dictates:

| `-ksp_norm_type` | what we feed to `(*ksp->converged)` and `KSPMonitor` | source in GMstab state |
|---|---|---|
| `KSP_NORM_NONE` | nothing — bypass convergence test, run to `max_it` | n/a |
| `KSP_NORM_UNPRECONDITIONED` | `‖b − Ax‖₂` | exactly `beta` for no-PC and right-PC; for left/split, recover via one extra `PCApply` |
| `KSP_NORM_PRECONDITIONED` | `‖M⁻¹(b − Ax)‖₂` (left), `‖A·M⁻¹(b − Ax)‖₂` (right) per PETSc convention | for left, `beta_pc` is already this; for right, apply `PCApply(M⁻¹, r)` and norm |
| `KSP_NORM_NATURAL` | M-energy norm | `PETSC_ERR_SUP` initially; can add later |

The validation harness uses `-ksp_norm_type unpreconditioned -ksp_atol 1e-10
-ksp_rtol 0` — strict mode, convergence on actual `‖b − Ax‖`. This is also
exactly what MATLAB `Solver_GMsStab.m` does (`iterres = norm(r)`,
`r = b − Ax`). So **for the unpreconditioned validation case, the C++/MATLAB
convergence test and PETSc's `unpreconditioned` mode are bit-identical by
construction.**

In the inner loop of `KSPSolve_GMSTAB`:

```c
PetscScalar normr;
switch (ksp->normtype) {
case KSP_NORM_UNPRECONDITIONED:
  if (ksp->pc_side == PC_LEFT) {
    PetscCall(VecNorm(work_unpc, NORM_2, &normr)); /* work_unpc = M·r_internal */
  } else {
    normr = gms->beta;
  }
  break;
case KSP_NORM_PRECONDITIONED:
  normr = gms->beta_pc;
  break;
case KSP_NORM_NONE:
  normr = 0;
  break;
default:
  SETERRQ(comm, PETSC_ERR_SUP, "norm type %s unsupported with KSPGMSTAB",
          KSPNormTypes[ksp->normtype]);
}
ksp->rnorm = normr;
PetscCall(KSPLogResidualHistory(ksp, normr));
PetscCall(KSPMonitor(ksp, ksp->its, normr));
PetscCall((*ksp->converged)(ksp, ksp->its, normr, &ksp->reason, ksp->cnvP));
if (ksp->reason) break;
```

Identical in shape to `KSPSolve_GMRES` (`src/ksp/ksp/impls/gmres/gmres.c:108-130`).
Audited pattern.

---

## 5. User-confirmed answers (2026-05-02)

| # | question | answer |
|---|---|---|
| 1 | branch strategy: `ksp-gmstab` from `amgx-block-aij-support` tip? | **YES** — done at `git checkout -b ksp-gmstab amgx-block-aij-support` |
| 2 | shadow space `P` API: `KSPGMSTABSetShadowSpace(KSP, Mat)` + `-ksp_gmstab_p_file <path>`? | **YES** — both interfaces |
| 3 | validation report format: per-baseline `petsc_residuals.csv` + rolled-up `validation_report.csv` matching existing schema? | **YES** — good set |
| 4 | recycling support: defer to Phase 8? | **YES** — phases 1-7 skip recycling, Phase 8 adds it |
| 5 | GPU validation: Phase 5a CPU bit-equivalence (≤1e-10), Phase 5b GPU smoke (≤1e-12)? | **YES** — 5a/5b split |

**Process decisions:**

- Atomic git commits per phase boundary, with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer matching the AMGX-fix style.
- Background-job completions (long sweep, AMGX repatch rerun, HDG top-30) are surfaced at the next phase boundary, not as interruptions. Continuous progress on GMstab is the priority.

---

## 6. Phase plan with verification gates

| phase | scope | days | gate |
|---|---|---|---|
| **0** | preflight: reproduce C++ baselines on current host, branch off `amgx-block-aij-support` | 0.5 | C++ port reproduces all 5 large + 10 of 124 small baselines |
| **1** | KSP type skeleton (registration, options, view, no-op solve) | 1.0 | `-ksp_type gmstab` registered, `-ksp_view` shows knobs, stub solve exits cleanly |
| **2** | inner modified GMRES (`pGMRESm` + `augGMRESm`) with component-level unit tests | 2.0 | unit tests vs C++ component-wise ≤1e-13 on `cdr_small` |
| **3** | Initialisation + cycle bodies (`gmstab1`, `gmstab2`) + flying restart driver | 3.0 | end-to-end no-PC convergence on `cdr_small` matches C++ to ≤1e-10 |
| **4a** | preconditioner side: right | 0.5 | `pc_type jacobi -ksp_pc_side right` smoke pass |
| **4b** | preconditioner side: left | 0.25 | `pc_type jacobi -ksp_pc_side left` smoke pass |
| **4c** | preconditioner side: symmetric | 0.25 | `pc_type icc -ksp_pc_side symmetric` smoke pass |
| **5a** | **CPU bit-equivalence validation, all 129 baselines, ≤1e-10 drift** | **2.0** | **129/129 PASS — load-bearing gate, no proceeding without** |
| **5b** | GPU validation (CUSPARSE), ≤1e-12 acceptable | 0.5 | smoke pass on cdr_small/sherman5/ocean GPU |
| **6** | sweep harness integration (~50 new GMstab × precond combinations) | 0.5 | run_one_matrix.sh produces gmstab rows on cdr_small |
| **7** | full sweep (5 matrices, ~580 configs) + HDG top-30 vs GMRES head-to-head | 1.0 (mostly compute) | comparison table, written-up summary |
| **8** | recycling support (`tolabs2`, `KSPGMSTABSetRecyclingSpace`) | 1.5 | smoke test + 1 multi-solve sequence test on time-stepping fixture |

**Total: ~13 working days, with Phase 5a as the load-bearing correctness gate.**

---

## 7. Phase 5 (validation) — detailed mechanism

This is the load-bearing checkpoint. We do NOT proceed past Phase 5a until
every baseline passes.

### 7.1 New entry-point binary

`src/ksp/ksp/tests/ex_gmstab_validate.c` (~250 LOC):
1. Loads `linsys.bin` (EXASIMLS format — reuse miniapp loader).
2. Loads deterministic `P.bin` (length N×s float64 little-endian, no header).
3. Configures `KSP`:
   ```
   -ksp_type gmstab -ksp_gmstab_s 4 -ksp_gmstab_L 2
   -pc_type none
   -ksp_atol 1e-10 -ksp_rtol 0 -ksp_max_it 500
   -ksp_norm_type unpreconditioned
   ```
4. Sets shadow space programmatically via `KSPGMSTABSetShadowSpace`.
5. Registers a custom `KSPMonitor` emitting CSV in the format
   `iter,matvec,iterres,trueres,runtime,runtime_mv` matching `cpp_residuals.csv`.

### 7.2 Validation script

`src/ksp/ksp/tests/gmstab_validate.py`:
- For each baseline directory: invokes `ex_gmstab_validate`, captures
  `petsc_residuals.csv`, diffs against `cpp_residuals.csv` and against
  MATLAB `residuals.csv`, reports per-column max absolute drift, asserts ≤1e-10.
- Output: `validation_report.csv` matching `validation_summary.csv` schema.

### 7.3 Verification matrix

| baseline | size (N) | runtime budget | PASS criterion |
|---|---|---|---|
| cdr_small | 729 | <1 s | per-iter `iterres` drift ≤1e-10, `matvec` exact |
| sherman5 | 3312 | <2 s | same |
| ocean | 143437 | <30 s | same |
| asic_320ks | 321671 | <60 s | same |
| torso1 | 116158 | <30 s | same |
| cdr_sweep_small × 124 | 729 each | <1 min total | same |

### 7.4 Known sources of acceptable drift

- **`MatTransposeMatMult` reduction order** between Eigen and PETSc small-dense
  kernels. At s=4, summing 4 products per element, reorder error is O(eps × s)
  ≈ 1e-15. Invisible in practice.
- **GPU non-deterministic reductions** (Phase 5b only). Looser ≤1e-12
  tolerance accepted.

### 7.5 Pure non-issues we can pre-empt

- **Random-number-generator drift from MATLAB.** The validation suite ships
  `P.bin` precisely to sidestep this. We never call our own
  `default_shadow_space` during validation; we always read MATLAB's exported
  `P`. RNG bit-equivalence becomes an issue when shipping the production
  solver but is separable from this verification.

---

## 8. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Small-dense sign convention drift (Eigen vs LAPACK) breaks bit-equivalence | medium | high | Phase 2 unit tests against C++ port at component level before integration |
| `MatTransposeMatMult` reduction order drift at >1e-12 | low | low | document in validation report; tolerance threshold absorbs it |
| `KSPSetUp` fires before user has set shadow space `P` (race in test infra) | medium | medium | both `KSPGMSTABSetShadowSpace` API + auto-create RNG fallback if neither set |
| `pc_side = KSP_PC_LEFT` validation has no MATLAB baseline | high (built-in) | low | Phase 5 only validates `pc_side = NONE`; left/right/split smoke-tested in Phase 4 against analytic problems |
| `KSPReset` not called on operator change → stale shadow space | low | medium | implement `KSPReset_GMSTAB` carefully; test via `KSPSetOperators` cycle |
| GMstab is slower than GMRES on the HDG matrix and the project's premise is undermined | unknown | conceptual, not technical | the comparison harness *is* the answer; this is what we're measuring |
| Inner GMRES `linsolve` on near-singular small dense | medium | medium | port the C++ port's "soft breakdown" handling (status flag + cycle restart); mirror MATLAB |
| Long sweep / HDG top-30 / AMGX repatch background jobs interfere with build | low | low | jobs use existing `libpetsc.so` via deleted-but-mapped fd; new branch builds independently |

---

## 9. Working git state at archival time

```
$ git log --oneline -3
ff2d3d288ff (HEAD -> ksp-gmstab, amgx-block-aij-support) amgx: support MATAIJ inputs that advertise bSize > 1
2d93faec18c amgx: make PCDestroy_AMGX defensive about partial initialization
7d1cbee15b2 (origin/main) Merge branch 'jolivet/configure-update-mumps' into 'main'
```

Working tree clean except untracked artifacts (`build_*.sh`,
`test_amgx_*_build/`) which are local-only and won't enter commits.

Local AMGX patch live in `arch-cuda-opt-i32/externalpackages/amgx-2.4.0/include/matrix.h:984`
(drop of `m_fixed_view_size` precondition in `getFixedSizesForView`). This patch is
NOT in PETSc git history — it's a hand-applied modification to the downloaded
AMGX source tree and will be re-applied if `--download-amgx` is re-run from
scratch. See `petsc/AMGX_BLOCKSIZE_BUG.md` for the upstream-submission writeup.

---

## 10. Sign-off

This plan is approved by Sam Austin on 2026-05-02 (per the conversation
trail leading to this archive). All 5 design questions have been answered
(§5). The next action after this archive lands is **Phase 0**: rebuild
`gmstab_cpp` from current `main` and reproduce all 5 large + 10 of 124
small baselines, then begin Phase 1.

If any of the assumptions in this plan turn out to be wrong during
implementation (e.g. `MatTransposeMatMult` introduces unacceptable drift,
or LAPACK sign conventions fight Eigen at >1e-13 component-wise drift),
we revisit and amend this archive before continuing.
