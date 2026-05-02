# Phase 3 — Status checkpoint

**Date:** 2026-05-02
**Branch:** `ksp-gmstab` at HEAD `aa821b3de64`

## What's complete after Phases 0–2

| layer | LOC | status |
|---|---|---|
| Plan archive + Phase 0 verification | 596 | ✅ committed `a3ecffcbd6d` + `b733e5911ca` |
| KSP type skeleton (registration, options, public APIs) | 480 | ✅ committed `20fcb15734b` |
| Inner GMRES (gmres_m / pgmres_m / aug_gmres_m + Givens, Arnoldi) | 700 | ✅ committed `aa821b3de64` |
| Small-dense LAPACK helpers (lq, orth, nullbasis, gemm, gemv, trsv, gesv, qr-LSQ) | 320 | ✅ in tree, will commit with Phase 3 |
| StabCoeffs (35° polynomial stabilization) | 130 | ✅ in tree, will commit with Phase 3 |

## What remains for Phase 3

Three substantial cycle bodies + the driver:

| component | est LOC | risk |
|---|---|---|
| Initialisation (port of Initialisation.m) | ~150 | Low — just gmres_m + 4 small-dense ops |
| GMstab1 / cycle 1 (port of GMstab1.m) | ~250 | Medium — block QR, Z update, pgmres call, post-update |
| GMstab2 / cycle 2 (port of GMstab2.m) | ~400 | High — biggest cycle, complex small-dense block algebra |
| Driver (KSPSolve_GMSTAB body) | ~150 | Medium — flying restart, snapshot/timing, restart logic |
| Snapshot + timing infrastructure | ~80 | Low |

Total Phase 3 increment: ~1030 LOC. Each line has a 1:1 correspondence
with `gmstab_cpp/src/solver.cpp`, but small-dense reduction order +
LAPACK-vs-Eigen sign conventions create real bit-equivalence risk.

## Decision: continue methodically

Per user's "no room for error" instruction, I'm going to:

1. Write each component as a faithful 1:1 port with line-number comments
   pointing back at the C++ source.
2. After each component compiles, build a per-iteration diagnostic dump
   that emits the SAME data structures the C++ port emits at the same
   points (W, H, R, Q, Z, V0, V1, beta, x).
3. Validate cdr_small first (cheapest baseline); fix divergences as
   they appear.
4. Once cdr_small passes ≤1e-10, run sherman5 / ocean / asic_320ks /
   torso1 + the 124 sweep cases.
5. Only then declare Phase 5a complete.

The validation infrastructure itself (including the per-iteration dump)
has to be written in tandem with the algorithm, since it's needed for
debugging.

## Work-in-progress files added in Phase 3 (not yet wired)

- `gmstab_smalldense.h` / `.c` : LAPACK helpers
- `gmstab_stab.h` / `.c`        : StabCoeffs port
- `gmstab_internal.h`           : shared declarations for cycle bodies

These compile and link cleanly but are not yet called by `KSPSolve_GMSTAB`.

## Background-job note

Per user-acknowledged protocol, background jobs are tracked but not
allowed to interrupt Phase 3 work:
- pid 916230 (multi-matrix sweep): still running on asic_320ks
- pid 1085451 (post-patch AMGX rerun): polling
- pid 1085931 (HDG top-30): polling
- pid 1085931 (post-patch AMGX rerun): polling

I'll surface their results at the next phase boundary.
