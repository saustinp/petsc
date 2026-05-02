/*
   Cross-file private header for KSPGMSTAB.

   Declares the cycle bodies (Initialisation / gmstab1 / gmstab2) and
   helper functions used by the outer KSPSolve_GMSTAB driver.
*/
#pragma once
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <../src/ksp/ksp/impls/gmstab/gmstabimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_modgmres.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_smalldense.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_stab.h>

/* Initialisation — port of Initialisation.m. Builds the initial V0, V1, Z
   (lower-triangular) plus updated x, r, beta from a cold start.

   Inputs (modified in place):
     ksp        : configured KSP (operator + PC are accessible)
     gms        : private state; gms->P is the shadow space, gms->s is set
     ws         : inner GMRES workspace, sized for m_max ≥ s
     x_local    : in/out solution vector; x_local += W * xi
     r          : in/out residual; r := W * c0
     beta       : in/out residual norm
*/
PETSC_INTERN PetscErrorCode KSPGMSTABInitialisation_Private(KSP ksp, KSP_GMSTAB *gms,
                                                             KSPGMSTABInnerWorkspace *ws,
                                                             Vec x_local, Vec r, PetscReal *beta);

/* GMstab1 — L=1 cycle. */
PETSC_INTERN PetscErrorCode KSPGMSTABCycle1_Private(KSP ksp, KSP_GMSTAB *gms,
                                                     KSPGMSTABInnerWorkspace *ws,
                                                     Vec x_local, Vec r, PetscReal *beta);

/* GMstab2 — L=2 cycle. */
PETSC_INTERN PetscErrorCode KSPGMSTABCycle2_Private(KSP ksp, KSP_GMSTAB *gms,
                                                     KSPGMSTABInnerWorkspace *ws,
                                                     Vec x_local, Vec r, PetscReal *beta);

/* Default shadow-space generator: P = orth(randn(N, s)) using std::mt19937_64.
   Port of gmstab_cpp's default_shadow_space. NOT bit-equivalent to MATLAB's
   `rng('default'); orth(randn(N,s))` (they use different state-traversal
   orders); only used for production runs where bit-equivalence with MATLAB
   isn't a concern. The validation harness should always use
   KSPGMSTABSetShadowSpace or -ksp_gmstab_p_file. */
PETSC_INTERN PetscErrorCode KSPGMSTABBuildDefaultShadow_Private(KSP ksp, KSP_GMSTAB *gms);

/* Load shadow space from a P.bin file (column-major doubles, no header). */
PETSC_INTERN PetscErrorCode KSPGMSTABLoadShadowFile_Private(KSP ksp, KSP_GMSTAB *gms);

/* Snapshot — emits a row to gms->trace_fp if tracing enabled, AND
   updates gms->matvec_count/snapshot_count/timing accumulators.

   x_total = xGlobal + x_local (the PerfMeasure::read convention).

   The extra MatMult to recompute the true residual is uncounted (MATLAB
   semantics). */
PETSC_INTERN PetscErrorCode KSPGMSTABSnapshot_Private(KSP ksp, KSP_GMSTAB *gms,
                                                       Vec x_total, PetscReal iter_norm);
