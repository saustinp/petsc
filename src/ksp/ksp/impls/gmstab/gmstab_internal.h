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

/* Like KSPGMSTABSnapshot_Private but takes x_local (the per-cycle update,
   typically reset to zero after each restart) and internally constructs
   x_total = gms->x_global + x_local. Use this at *every* snapshot site
   where you have x_local in scope and the cycle/driver may have
   accumulated a non-zero x_global from prior restarts. The pgmres /
   aug_gmres mid-iter callbacks already build x_global_plus_local
   themselves (they need to evaluate at intermediate iter solutions);
   they should keep using KSPGMSTABSnapshot_Private directly. */
PETSC_INTERN PetscErrorCode KSPGMSTABSnapshotLocal_Private(KSP ksp, KSP_GMSTAB *gms,
                                                            Vec x_local, PetscReal iter_norm);

/* Phase 4a — finalize the user-visible solution before returning from
   KSPSolve_GMSTAB. Walks the algorithm-internal x_alg = x_global + x_local
   into the user's algebra, applying the inverse of any right/symmetric
   preconditioning. Specifically:

     pc_side = PC_NONE  or PC_LEFT   →  x_local := x_global + x_local
     pc_side = PC_RIGHT              →  x_local := x_initial_guess + B⁻¹ ·
                                                  (x_global + x_local − x_initial_guess)
     pc_side = PC_SYMMETRIC          →  x_local := x_initial_guess + B_R⁻¹ ·
                                                    (x_global + x_local − x_initial_guess)
                                        via PCApplySymmetricRight (Phase 4c). Only the
                                        right factor B_R⁻¹ is applied — not full PCApply
                                        which would be (B_L·B_R)⁻¹ in symmetric splits
                                        and produce wrong x_user. Algorithm pre-applies
                                        B_L⁻¹ to bLocal once at solve start; symmetry
                                        of the algebra means finalize undoes only the
                                        right factor. Requires the user's PC type to
                                        implement applysymmetricright (PCJACOBI,
                                        PCICC, PCBJACOBI(with right sub-PC), etc.;
                                        not PCSOR/PCASM/PCGAMG/PCHYPRE).

   Must be called from EVERY return path in KSPSolve_GMSTAB that returns
   AFTER `VecSet(x_local, 0.0)` has zeroed the initial guess in x_local.
   Idempotent for PC_NONE/PC_LEFT; for PC_RIGHT it does the unwrap in place
   on x_local. Callers must not VecAXPY x_global into x_local separately —
   this helper subsumes that. */
PETSC_INTERN PetscErrorCode KSPGMSTABFinalizeSolution_Private(KSP ksp, KSP_GMSTAB *gms,
                                                               Vec x_local);
