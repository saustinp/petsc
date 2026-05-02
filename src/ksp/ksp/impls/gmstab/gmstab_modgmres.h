/*
   Private header for KSPGMSTAB inner GMRES variants.

   Implements three inner-loop solvers used by the GMstab cycles, all
   sharing the same Arnoldi + Givens-on-the-fly skeleton but differing
   in the projection step that wraps the matvec:

     gmres_m     :  W(:,i+1) := A * W(:,i)
                    (used in Initialisation only)

     pgmres_m    :  xi = Z \ (P' * W(:,i)),  W(:,i+1) := A * (W(:,i) - V0*xi)
                    (used in GMstab1, the L=1 cycle)

     aug_gmres_m :  W(:,i+1) := A * W(:,i),   xi = Z \ (P' * W(:,i+1));
                    W(:,i+1) -= V1 * xi
                    (used in GMstab2, the L=2 cycle)

   All three use CLASSICAL Gram-Schmidt for orthogonalisation (NOT modified
   GS) — this matches the MATLAB reference exactly; switching to modified
   GS would produce a different Hessenberg in finite precision and break
   the bit-equivalence acceptance criterion (Phase 5a, ≤ 1e-10 drift).

   The Givens rotations on the upper-Hessenberg-to-upper-triangular
   reduction use LAPACK DLARFG sign convention exactly:
       beta = -sign(a) * hypot(a, b),
   matching MATLAB's `qr` and the C++ port's `givens_qr_2x1`.
*/
#pragma once
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <../src/ksp/ksp/impls/gmstab/gmstabimpl.h>

/* 2x1 Givens reflector in the LAPACK DLARFG convention.
   `Q` is the symmetric orthogonal 2x2 matrix such that Q * [beta; 0] = [a; b].
   Identity case (b == 0): Q = I, beta = a.
   Otherwise: beta = -copysign(hypot(a,b), a). */
typedef struct {
  PetscScalar Q[2][2];
  PetscScalar beta_val;
} PetscGivens2x1;

/* Compute the Givens rotation that triangularises the 2x1 column [a; b]. */
PETSC_INTERN void KSPGMSTABGivens2x1_Private(PetscScalar a, PetscScalar b, PetscGivens2x1 *g);

/* Apply g to a 2-element strip in place: [a, b] <- Q * [a, b]. */
static inline void KSPGMSTABApplyGivens2_Private(const PetscGivens2x1 *g, PetscScalar *a, PetscScalar *b)
{
  const PetscScalar new_a = g->Q[0][0] * (*a) + g->Q[0][1] * (*b);
  const PetscScalar new_b = g->Q[1][0] * (*a) + g->Q[1][1] * (*b);
  *a = new_a;
  *b = new_b;
}

/* ============================================================================
   Common workspace for the inner GMRES variants. Allocated per-call from the
   KSPGMSTAB scratch arena; sized for the largest of (s, 2s+2) so a single
   workspace serves all three variants.
   ============================================================================ */
typedef struct {
  PetscInt     s;          /* shadow-space dimension */
  PetscInt     m;          /* number of Arnoldi steps THIS call */

  /* W is the Krylov basis MatDense (N x (m+1)). Owned externally — caller
     pre-allocates and passes the same handle on every call to avoid malloc
     churn. We only write into columns 0..m of W, so callers can pass W of
     size N x (m_max+1) where m_max >= m. */
  Mat          W;

  /* Y is small dense, host-side, column-major. Sized s x (m+1) — the worst
     case (used by pgmres). aug_gmres uses only the first m columns. */
  PetscScalar *Y;
  PetscInt     Y_cols;     /* allocated column count (>= m+1) */

  /* H, R: (m+1) x m, column-major. R may have one extra row (the
     to-be-zeroed entry below the diagonal) — we keep it (m+1) x m
     for clean rotation indexing. */
  PetscScalar *H;
  PetscScalar *R;

  /* Q: (m+1) x (m+1), column-major. Identity-initialised, then rotated by
     all m Givens. The MATLAB / C++ port uses Q'(0..m-1, 0..m).T as the
     output Q — we emit it in the same convention. */
  PetscScalar *Q;

  /* gamma: m+1 vector, the rotated RHS in the least-squares problem. */
  PetscScalar *gamma;

  /* Stored Givens rotations, m of them. */
  PetscGivens2x1 *G;

  /* Length-N scratch (for matvec output before column copy if needed). */
  Vec work_n;
} KSPGMSTABInnerWorkspace;

/* Allocate / free the inner workspace. Caller passes m_max = max Arnoldi
   steps any single call may take (= max(s, 2s+2) = 2s+2 in practice). */
PETSC_INTERN PetscErrorCode KSPGMSTABInnerWorkspaceCreate_Private(KSP ksp, PetscInt s, PetscInt m_max,
                                                                  Vec template_vec,
                                                                  KSPGMSTABInnerWorkspace *ws);
PETSC_INTERN PetscErrorCode KSPGMSTABInnerWorkspaceDestroy_Private(KSPGMSTABInnerWorkspace *ws);

/* Per-snapshot callback fired by pgmres / aug_gmres at the bottom of each
   inner iteration with (current x_local + xGlobal, current iter-residual
   norm). Returning PETSC_TRUE from `should_terminate` causes the inner
   solver to exit early in the same way the MATLAB reference does. */
typedef PetscErrorCode (*KSPGMSTABInnerSnapshot)(KSP ksp, Vec x_global_plus_local, PetscReal iter_norm,
                                                 PetscBool *should_terminate, void *ctx);

/* ============================================================================
   The three inner-loop variants. Each returns:
     - x_in_out  : updated x   (x_in + W * zeta + projection terms)
     - r_in_out  : updated r   (W * (gamma_full - H*zeta))
     - beta_out  : ||r_in_out|| (the iter-residual norm)
     - converged_inside : did the inner GMRES converge to tolabs before m steps?

   Workspace fields (W, Y, H, Q, R, gamma, G) are populated and accessible
   to the caller after the call returns; they're the inputs the cycle
   bodies need for their post-inner-solve linear-algebra updates.
   ============================================================================ */

/* gmres_m. Used in Initialisation. m = s typical. */
PETSC_INTERN PetscErrorCode KSPGMSTABGmresM_Private(KSP ksp,
                                                    Vec x_in, Vec r_in, PetscReal beta_in,
                                                    PetscReal tolabs,
                                                    KSPGMSTABInnerWorkspace *ws,
                                                    Vec x_out, Vec r_out, PetscReal *beta_out,
                                                    PetscBool *converged_inside);

/* pgmres_m. Used in GMstab1. P is N x s, Z is s x s lower-tri host buffer
   (column-major), V0 is N x s MatDense. m = s. */
PETSC_INTERN PetscErrorCode KSPGMSTABPGmresM_Private(KSP ksp,
                                                     Mat P, const PetscScalar *Z, PetscInt Z_ldim,
                                                     Mat V0,
                                                     Vec x_in, Vec r_in, PetscReal beta_in,
                                                     PetscReal tolabs,
                                                     KSPGMSTABInnerWorkspace *ws,
                                                     Vec x_global,
                                                     KSPGMSTABInnerSnapshot snap_cb, void *snap_ctx,
                                                     Vec x_out, Vec r_out, PetscReal *beta_out,
                                                     PetscBool *converged_inside,
                                                     PetscBool *terminated_early);

/* aug_gmres_m. Used in GMstab2. V0 is N x s, V1 is N x s. m = 2s+2. */
PETSC_INTERN PetscErrorCode KSPGMSTABAugGmresM_Private(KSP ksp,
                                                       Mat P, const PetscScalar *Z, PetscInt Z_ldim,
                                                       Mat V0, Mat V1,
                                                       Vec x_in, Vec r_in, PetscReal beta_in,
                                                       PetscReal tolabs,
                                                       KSPGMSTABInnerWorkspace *ws,
                                                       Vec x_global,
                                                       KSPGMSTABInnerSnapshot snap_cb, void *snap_ctx,
                                                       Vec x_out, Vec r_out, PetscReal *beta_out,
                                                       PetscBool *converged_inside);
