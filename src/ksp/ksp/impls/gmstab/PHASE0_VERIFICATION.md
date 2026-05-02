# Phase 0 — Preflight verification report

**Date:** 2026-05-02
**Branch:** `ksp-gmstab` at commit `a3ecffcbd6d` (parent: `amgx-block-aij-support` `ff2d3d288ff`)
**Goal:** Confirm the C++ port reproduces every bit-for-bit baseline today on
the current host. Establishes the oracle that Phase 5a will validate against.

## 1. C++ binary build state

`/home/sam/hpc_stack/gmstab_cpp/build/gmstab_run` was compiled 2026-05-01 02:01,
linking `libgmstab.a` of the same date. CMake configuration uses Eigen at
`/home/sam/.local/amr/include/eigen3` (verified existing). No rebuild required.

## 2. Large-baseline reproduction

For each baseline `<name>` in `{cdr_small, sherman5, ocean, asic_320ks, torso1}`,
ran:

```bash
gmstab_run --linsys baselines/<name>/linsys.bin \
           --P_bin   baselines/<name>/P.bin \
           --s 4 --tol 1e-10 --maxmatvec 500 --maxruntime 120 \
           --out /tmp/gmstab_phase0/<name>_residuals.csv
```

**Result: 5/5 PASS — every test reproduces bit-exactly.**

| baseline | rows | matvec match | max iterres drift | max trueres drift |
|---|---|---|---|---|
| cdr_small | 233 | exact | 0.000e+00 | 0.000e+00 |
| sherman5 | 618 | exact | 0.000e+00 | 0.000e+00 |
| ocean | 618 | exact | 0.000e+00 | 0.000e+00 |
| asic_320ks | 18 | exact | 0.000e+00 | 0.000e+00 |
| torso1 | 617 | exact | 0.000e+00 | 0.000e+00 |

## 3. Sweep-baseline spot check

Ten of 124 `cdr_sweep_small/*` directories selected at random, with per-case
`--maxmatvec` and `--maxruntime` parsed from each `summary.txt`:

**Result: 10/10 PASS bit-exactly.**

```
PASS eps_10_beta_1_r_1: rows=53 drift=0.00e+00 mvc_ok=True
PASS eps_0_beta_1000_r_0: rows=193 drift=0.00e+00 mvc_ok=True
PASS eps_0.01_beta_1_r_1: rows=135 drift=0.00e+00 mvc_ok=True
PASS eps_1_beta_0_r_0: rows=29 drift=0.00e+00 mvc_ok=True
PASS eps_0.01_beta_10_r_1: rows=244 drift=0.00e+00 mvc_ok=True
PASS eps_0_beta_10_r_0: rows=173 drift=0.00e+00 mvc_ok=True
PASS eps_0.01_beta_100_r_1: rows=247 drift=0.00e+00 mvc_ok=True
PASS eps_0_beta_0_r_10: rows=3 drift=0.00e+00 mvc_ok=True
PASS eps_0_beta_100_r_10: rows=203 drift=0.00e+00 mvc_ok=True
PASS eps_0.1_beta_1000_r_100: rows=255 drift=0.00e+00 mvc_ok=True
```

## 4. Critical insight: per-case parameters matter

**The first spot-check pass FAILED on `eps_0.01_beta_10_r_1` with drift 4.31e-3
when invoked with `--maxmatvec 500`.** Investigation revealed the cause: that
sweep case was generated with `maxmatvec: 200`. The GMstab algorithm makes
internal decisions (flying-restart triggers, cycle-type selection, when to
enter pGMRESm vs aug_gmres_m) based on the runtime budget. Running past the
generation budget produces a different trajectory.

**Implication for Phase 5a:** the validation harness MUST read each baseline's
`summary.txt` and pass the recorded `maxmatvec` / `maxruntime` / `tolabs` as
`-ksp_max_it` / a custom timeout / `-ksp_atol` to the PETSc port.

## 5. Source-of-truth files read end-to-end (Phase 0 prep)

In preparation for Phase 1+ implementation, the entire C++ port was read with
detailed annotation:

- `include/gmstab/types.hpp` — column-major Eigen dense, row-major sparse CSR
- `include/gmstab/csr_matrix.hpp` — `LinOp = std::function<Vec(const Vec&)>`
- `include/gmstab/small_dense.hpp` + `src/small_dense.cpp` — `orth` (SVD-based
  to match MATLAB), `roworth`, `nullbasis` (Householder QR full Q),
  `lq` ([Q,L] = (qr(Z'))' style, L lower-tri), `dir_rbio` (the
  Z\eta + V0/V1 update), `givens_qr_2x1` (LAPACK DLARFG sign convention)
- `include/gmstab/perf_measure.hpp` + `src/perf_measure.cpp` — counted matvec,
  paused-timer `read()` that does an extra uncounted true-residual matvec,
  termination via `tolabs` || `maxmatvec` || `maxruntime`
- `include/gmstab/stab_coeffs.hpp` + `src/stab_coeffs.cpp` — 35° angle
  (`alpha = pi*7/36`), forced-angle "maintaining-the-convergence" branch,
  `beta_sq < 0` rescue path (rebuilds `S(0,0)` from `||r(:,0)||^2`)
- `include/gmstab/modified_gmres.hpp` + `src/modified_gmres.cpp` — three
  variants `gmres_m`, `pgmres_m`, `aug_gmres_m`, all using **classical**
  (NOT modified) Gram-Schmidt to match MATLAB; Givens rotations stored as
  `std::vector<Givens2x1>`; `apply_inner_givens` helper
- `include/gmstab/solver.hpp` + `src/solver.cpp` — `solve()` driver with
  flying-restart logic (`cRestart=1e-2`, `cReplace=1e-2`, `n2cyclesMax=3`),
  `Initialisation`, `gmstab1` (L=1 cycle), `gmstab2` (L=2 cycle)

## 6. Algorithm-level decisions captured for the PETSc port

### 6.1 Solver_GMsStab.m → KSPSolve_GMSTAB driver

Top-level driver structure to replicate exactly:

```
1.  Build P (or accept programmatic override / file load)
2.  Resume timer
3.  bLocal = b - A * xGlobal      [first counted matvec]
4.  r = bLocal, x = 0, beta = ||r||, betaLocal = beta
5.  Initialisation (or recycling restart)
6.  perf.read(xGlobal + x, beta)
7.  betaMax = max(beta, betaLocal)
8.  while beta > tolabs:
        if beta < 1e-2 * betaLocal: t_restart = true
        elif beta < 1e-2 * betaMax: t_replace = true
        else: betaMax = max(betaMax, beta)
        L = 1 if (t_restart || t_replace || n2cycles > 3) else 2
        if L == 1: gmstab1(...);   n2cycles = 0
        else:      gmstab2(...);   n2cycles += 1
        [optional: hU recycling — Phase 8]
        if beta <= tolabs: break
        if t_restart || t_replace:
            print "restart"
            r = bLocal - A * x         [counted matvec]
            eta = P' * r
            dir_rbio(V0, V1, Z, x, r, eta)
            beta = ||r||;  betaMax = beta
        else: betaMax = max(beta, betaMax)
        if t_restart: bLocal=r; xGlobal+=x; x=0; betaLocal=beta
        perf.read(xGlobal + x, beta)
        if perf.isTerminate(): break
```

### 6.2 Initialisation.m → KSPGMSTABInitialise_Private

```
1. (W, H, Q, R) ← gmres_m(A, x0, r0, beta, m=s)        [s counted matvecs]
2. Y = P' * W                  [s × (s+1)]
3. eta = beta * Y(:, 0)        [s]
4. Z = Y * Q                   [s × s]
5. (Q_z, L_z) = lq(Z)          [Z = L_z * Q_z']
6. xi = R \ (Q_z * (L_z \ eta))
7. gamma_vec = [beta; 0; ...; 0]    [(s+1)]
8. c_0 = gamma_vec - H * xi
9. Z := L_z (lower-triangular)
10. x ← x_in + W(:, 1..s) * xi
11. r ← W(:, 1..s+1) * c_0
12. beta = ||c_0||
13. V0 = W(:, 1..s) * (R \ Q_z)
14. V1 = W(:, 1..s+1) * (Q * Q_z)
```

### 6.3 GMstab1.m / GMstab2.m bodies

To be ported in detail in Phase 3 (their structures are large but
well-commented in the C++ port; this archive contains the line-by-line
correspondences via the file annotations above).

## 7. State of background jobs

| pid | job | status | duration |
|---|---|---|---|
| 916230 | multi-matrix long sweep | running, on `asic_320ks` (matrix 4/5) | 10h+ |
| 1085451 | post-patch AMGX rerun | polling for longrun completion | 1h+ |
| 1085931 | HDG top-30 selection + run | polling for longrun completion | 1h+ |

Per the user's instruction, these are surfaced at phase boundaries only
and do not interrupt GMstab work. They use the existing `libpetsc.so` and
do not interact with the `ksp-gmstab` branch.

## 8. Phase 0 status

**COMPLETE.** Oracle confirmed reproducible at bit-exact level across
5 large + 10 of 124 sweep baselines. Source-of-truth read end-to-end with
key invariants captured. Ready to begin Phase 1.
