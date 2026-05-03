/*
   Inner GMRES variants for KSPGMSTAB. See gmstab_modgmres.h for the
   contract and C++ reference correspondence.

   Implementation strategy:

   - W is a PETSc MatDense of shape N x (m+1), so individual basis columns
     are obtained as Vec views via MatDenseGetColumnVecRead / Write.
     This lets MatMult-style matvecs go directly through cuSPARSE on the
     GPU when the operator matrix is MATAIJCUSPARSE, with no host shuttle.

   - H, R, Q, gamma are small dense host buffers (column-major). The
     largest of them (Q) is at most (m+1)x(m+1) = (2s+3)x(2s+3) for
     aug_gmres with default s=4 — that's an 11x11 dense, fits in cache.

   - Givens rotations are stored as a strip of PetscGivens2x1 records and
     applied on the fly to (R[i,i], R[i+1,i]), to gamma's matching
     two-element strip, and to Q's two rows.

   - All three variants use CLASSICAL Gram-Schmidt (matching the C++ port
     and MATLAB), implemented as separate VecDot calls per j (BLAS-1
     style, NOT a fused VecMDot — this reproduces the reduction order
     that the MATLAB reference uses).
*/
#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_modgmres.h>
#include <petscblaslapack.h>
#include <math.h>

/* ============================================================================
 * Givens rotation construction — LAPACK DLARFG sign convention.
 * Mirrors gmstab_cpp/src/small_dense.cpp:givens_qr_2x1 exactly.
 * ============================================================================ */
PETSC_INTERN void KSPGMSTABGivens2x1_Private(PetscScalar a, PetscScalar b, PetscGivens2x1 *g)
{
  if (b == 0.0) {
    g->Q[0][0] = 1.0; g->Q[0][1] = 0.0;
    g->Q[1][0] = 0.0; g->Q[1][1] = 1.0;
    g->beta_val = a;
    return;
  }
  /* beta = -copysign(hypot(a, b), a). */
  const PetscReal h = hypot(PetscRealPart(a), PetscRealPart(b));
  const PetscReal sgn_a = (PetscRealPart(a) >= 0.0) ? 1.0 : -1.0;
  const PetscScalar beta_val = -sgn_a * h;

  const PetscScalar v0  = 1.0;
  const PetscScalar v1  = b / (a - beta_val);
  const PetscScalar tau = (beta_val - a) / beta_val;

  g->Q[0][0] = 1.0 - tau * v0 * v0;
  g->Q[0][1] = -tau * v0 * v1;
  g->Q[1][0] = -tau * v1 * v0;
  g->Q[1][1] = 1.0 - tau * v1 * v1;
  g->beta_val = beta_val;
}

/* ============================================================================
 * Inner workspace allocation / freeing.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABInnerWorkspaceCreate_Private(KSP ksp, PetscInt s, PetscInt m_max,
                                                                  Vec template_vec,
                                                                  KSPGMSTABInnerWorkspace *ws)
{
  PetscFunctionBegin;
  PetscCheck(s >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "s = %" PetscInt_FMT " must be >= 1", s);
  PetscCheck(m_max >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "m_max = %" PetscInt_FMT " must be >= 1", m_max);

  ws->s = s;
  ws->m = 0;
  /* Defensive: callers MUST set matvec_count_ptr immediately after this
     create call to enable matvec counting; until they do, NULL means
     "don't count" (gmres_m / pgmres_m / aug_gmres_m all check before
     dereferencing). Zeroing here protects against a future caller that
     forgets to set it — the alternative (uninitialised stack) would
     either crash or worse, silently dereference garbage memory. */
  ws->matvec_count_ptr = NULL;

  /* Build W as a MATDENSE that mirrors template_vec's layout. */
  PetscInt n_local;
  PetscCall(VecGetLocalSize(template_vec, &n_local));
  PetscInt N_global;
  PetscCall(VecGetSize(template_vec, &N_global));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &ws->W));
  PetscCall(MatSetType(ws->W, MATDENSE));
  PetscCall(MatSetSizes(ws->W, n_local, PETSC_DECIDE, N_global, m_max + 1));
  PetscCall(MatSetUp(ws->W));

  ws->Y_cols = m_max + 1;
  PetscCall(PetscCalloc1((size_t)s * (size_t)(m_max + 1), &ws->Y));

  PetscCall(PetscCalloc1((size_t)(m_max + 1) * (size_t)m_max, &ws->H));
  PetscCall(PetscCalloc1((size_t)(m_max + 1) * (size_t)m_max, &ws->R));
  PetscCall(PetscCalloc1((size_t)(m_max + 1) * (size_t)(m_max + 1), &ws->Q));
  PetscCall(PetscCalloc1((size_t)(m_max + 1), &ws->gamma));
  PetscCall(PetscCalloc1((size_t)m_max, &ws->G));

  PetscCall(VecDuplicate(template_vec, &ws->work_n));
  PetscCall(VecDuplicate(template_vec, &ws->work_in));
  PetscCall(VecDuplicate(template_vec, &ws->work_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABInnerWorkspaceDestroy_Private(KSPGMSTABInnerWorkspace *ws)
{
  PetscFunctionBegin;
  if (!ws) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&ws->W));
  PetscCall(PetscFree(ws->Y));
  PetscCall(PetscFree(ws->H));
  PetscCall(PetscFree(ws->R));
  PetscCall(PetscFree(ws->Q));
  PetscCall(PetscFree(ws->gamma));
  PetscCall(PetscFree(ws->G));
  PetscCall(VecDestroy(&ws->work_n));
  PetscCall(VecDestroy(&ws->work_in));
  PetscCall(VecDestroy(&ws->work_out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Reset the in-place state so a fresh inner solve has clean H/R/Q/gamma/G.
 * (Y is also zeroed since aug_gmres only touches m of its columns.)
 * ============================================================================ */
static PetscErrorCode KSPGMSTABInnerWorkspaceReset_Private(KSPGMSTABInnerWorkspace *ws, PetscInt m)
{
  PetscFunctionBegin;
  ws->m = m;

  const size_t Hsz = (size_t)(m + 1) * (size_t)m;
  const size_t Qsz = (size_t)(m + 1) * (size_t)(m + 1);
  const size_t Ysz = (size_t)ws->s * (size_t)(m + 1);

  PetscCall(PetscArrayzero(ws->H, Hsz));
  PetscCall(PetscArrayzero(ws->R, Hsz));
  PetscCall(PetscArrayzero(ws->Q, Qsz));
  PetscCall(PetscArrayzero(ws->Y, Ysz));
  PetscCall(PetscArrayzero(ws->gamma, m + 1));
  PetscCall(PetscArrayzero(ws->G, m));

  /* Q := identity (m+1) x (m+1) */
  for (PetscInt i = 0; i <= m; ++i) ws->Q[(size_t)i + (size_t)i * (size_t)(m + 1)] = 1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * apply_inner_givens — common Givens-update body used by all three variants.
 *
 *   1. Copy H[0..i+1, i] into R[0..i+1, i].
 *   2. Apply previously-stored G[0..i-1] to the (R[j,i], R[j+1,i]) strips.
 *   3. Compute G[i] from (R[i,i], R[i+1,i]); update R[i,i] = G[i].beta_val,
 *      R[i+1,i] = 0.
 *   4. Apply G[i] to (gamma[i], gamma[i+1]).
 *   5. Apply G[i] to Q's rows i and i+1 (across all (m+1) columns).
 * ============================================================================ */
static PetscErrorCode KSPGMSTABApplyInnerGivens_Private(KSPGMSTABInnerWorkspace *ws, PetscInt i)
{
  PetscFunctionBegin;
  const PetscInt m   = ws->m;
  const PetscInt ldH = m + 1;        /* leading dim of H, R (col-major) */
  const PetscInt ldQ = m + 1;        /* leading dim of Q (col-major) */

  /* (1) Copy H column i into R column i, rows 0..i+1. */
  for (PetscInt k = 0; k <= i + 1; ++k) {
    ws->R[(size_t)k + (size_t)i * (size_t)ldH] = ws->H[(size_t)k + (size_t)i * (size_t)ldH];
  }
  /* (2) Apply stored G[0..i-1] to (R[j,i], R[j+1,i]). */
  for (PetscInt j = 0; j < i; ++j) {
    PetscScalar *aij = &ws->R[(size_t)j     + (size_t)i * (size_t)ldH];
    PetscScalar *bij = &ws->R[(size_t)(j+1) + (size_t)i * (size_t)ldH];
    KSPGMSTABApplyGivens2_Private(&ws->G[j], aij, bij);
  }
  /* (3) New Givens. */
  PetscScalar a_ii = ws->R[(size_t)i     + (size_t)i * (size_t)ldH];
  PetscScalar a_ip = ws->R[(size_t)(i+1) + (size_t)i * (size_t)ldH];
  KSPGMSTABGivens2x1_Private(a_ii, a_ip, &ws->G[i]);
  ws->R[(size_t)i     + (size_t)i * (size_t)ldH] = ws->G[i].beta_val;
  ws->R[(size_t)(i+1) + (size_t)i * (size_t)ldH] = 0.0;

  /* (4) Apply G[i] to gamma[i..i+1]. */
  KSPGMSTABApplyGivens2_Private(&ws->G[i], &ws->gamma[i], &ws->gamma[i+1]);

  /* (5) Apply G[i] to Q's rows i and i+1. The C++ port stores Q row-by-row
     (Eigen's row-major access via Q.row()), but we keep Q column-major;
     "row" k means Q[k + j*ldQ] for j = 0..m. */
  for (PetscInt j = 0; j <= m; ++j) {
    PetscScalar *qa = &ws->Q[(size_t)i     + (size_t)j * (size_t)ldQ];
    PetscScalar *qb = &ws->Q[(size_t)(i+1) + (size_t)j * (size_t)ldQ];
    KSPGMSTABApplyGivens2_Private(&ws->G[i], qa, qb);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Helper: compute zeta = R[0..i, 0..i]^-1 * gamma[0..i] (upper-tri solve).
 * R is column-major, leading dim m+1. Result lands in zeta[0..i].
 * ============================================================================ */
static PetscErrorCode KSPGMSTABTriUSolve_Private(const PetscScalar *R, PetscInt ldR,
                                                  const PetscScalar *rhs, PetscScalar *zeta,
                                                  PetscInt n)
{
  PetscFunctionBegin;
  PetscBLASInt N_, ldR_, incx = 1;
  PetscCall(PetscBLASIntCast(n, &N_));
  PetscCall(PetscBLASIntCast(ldR, &ldR_));

  /* Copy rhs into zeta (BLAS trsv overwrites in place). */
  for (PetscInt k = 0; k < n; ++k) zeta[k] = rhs[k];

  /* trsv(uplo='U', trans='N', diag='N', n, R, ldR, x, incx)
     Solves R(0:n,0:n) * x = b in place. */
  PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "N", &N_, R, &ldR_, zeta, &incx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Lower-triangular solve. */
static PetscErrorCode KSPGMSTABTriLSolve_Private(const PetscScalar *L, PetscInt ldL,
                                                  const PetscScalar *rhs, PetscScalar *eta,
                                                  PetscInt n)
{
  PetscFunctionBegin;
  PetscBLASInt N_, ldL_, incx = 1;
  PetscCall(PetscBLASIntCast(n, &N_));
  PetscCall(PetscBLASIntCast(ldL, &ldL_));
  for (PetscInt k = 0; k < n; ++k) eta[k] = rhs[k];
  PetscCallBLAS("BLAStrsv", BLAStrsv_("L", "N", "N", &N_, L, &ldL_, eta, &incx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Helper: x_out = x_in + W[:, 0..n-1] * zeta. Computes the standard GMRES
 * correction "x = x0 + W * (R \ gamma)" given `n` Arnoldi vectors and the
 * already-solved zeta vector.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABAxpyKrylov_Private(Vec x_out, Vec x_in,
                                                   Mat W, const PetscScalar *zeta, PetscInt n)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x_in, x_out));
  for (PetscInt k = 0; k < n; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(W, k, &wk));
    PetscCall(VecAXPY(x_out, zeta[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(W, k, &wk));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Helper: r_out = W[:, 0..n] * (gamma_top - H[0..n+1, 0..n] * zeta).
 *
 * `gamma_top` is a length-(n+1) vector holding [beta_in, 0, ..., 0]; the
 * caller supplies it as a small dense buffer because the H slice we need
 * is the original (unrotated) Hessenberg, not R.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABResidualReconstruct_Private(Vec r_out,
                                                            Mat W,
                                                            const PetscScalar *H, PetscInt ldH,
                                                            const PetscScalar *zeta,
                                                            PetscReal beta_in,
                                                            PetscInt n)
{
  PetscFunctionBegin;
  /* Compute c = gamma_top - H[0..n+1, 0..n] * zeta into a small host buffer. */
  PetscScalar *c;
  PetscCall(PetscMalloc1(n + 1, &c));
  c[0] = (PetscScalar)beta_in;
  for (PetscInt k = 1; k <= n; ++k) c[k] = 0.0;

  /* c -= H * zeta. */
  PetscBLASInt M_, N_, ldH_, inc = 1;
  PetscScalar  alpha = -1.0, beta_  = 1.0;
  PetscCall(PetscBLASIntCast(n + 1, &M_));
  PetscCall(PetscBLASIntCast(n,     &N_));
  PetscCall(PetscBLASIntCast(ldH,   &ldH_));
  PetscCallBLAS("BLASgemv", BLASgemv_("N", &M_, &N_, &alpha, H, &ldH_, zeta, &inc, &beta_, c, &inc));

  /* r_out = W[:, 0..n] * c. */
  PetscCall(VecSet(r_out, 0.0));
  for (PetscInt k = 0; k <= n; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(W, k, &wk));
    PetscCall(VecAXPY(r_out, c[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(W, k, &wk));
  }
  PetscCall(PetscFree(c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Compute Y[:, i] = P^T * W[:, i], with the result deposited into the
 * column-major Y buffer at column `i`. P is a tall-skinny MATDENSE
 * (N x s); the multiply is one parallel reduction.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABProjectColumn_Private(Mat P, Vec wcol,
                                                     PetscScalar *Y, PetscInt s, PetscInt i, Vec scratch_s)
{
  PetscFunctionBegin;
  /* Strategy: VecDot s times — one VecDot per column of P. This matches the
     reduction order of the C++/Eigen `P.transpose() * w` (which Eigen
     implements as s separate dot products in column order). Using
     MatMultTranspose would also work but might reorder reductions
     internally. */
  for (PetscInt k = 0; k < s; ++k) {
    Vec pk;
    PetscScalar yk;
    PetscCall(MatDenseGetColumnVecRead(P, k, &pk));
    PetscCall(VecDot(wcol, pk, &yk));
    PetscCall(MatDenseRestoreColumnVecRead(P, k, &pk));
    Y[(size_t)k + (size_t)i * (size_t)s] = yk;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Standard Arnoldi step (used by gmres_m and the post-projection part of
 * pgmres / aug_gmres):
 *
 *   For j = 0 .. i:
 *       H[j, i] = W[:, j]^T * W[:, i+1]
 *       W[:, i+1] -= H[j, i] * W[:, j]
 *   H[i+1, i] = ||W[:, i+1]||
 *   W[:, i+1] /= H[i+1, i]
 *
 * NB: classical Gram-Schmidt, not modified — matches MATLAB.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABArnoldiStep_Private(KSPGMSTABInnerWorkspace *ws, PetscInt i)
{
  PetscFunctionBegin;
  Mat W = ws->W;
  const PetscInt ldH = ws->m + 1;

  /* Copy W[:, i+1] into work_out to operate on without holding a column-vec
     borrow (which would conflict with reading W[:, j] columns below). */
  {
    Vec wip1;
    PetscCall(MatDenseGetColumnVec(W, i + 1, &wip1));
    PetscCall(VecCopy(wip1, ws->work_out));
    PetscCall(MatDenseRestoreColumnVec(W, i + 1, &wip1));
  }

  for (PetscInt j = 0; j <= i; ++j) {
    Vec wj;
    PetscScalar hji;
    PetscCall(MatDenseGetColumnVecRead(W, j, &wj));
    PetscCall(VecDot(ws->work_out, wj, &hji));
    PetscCall(VecAXPY(ws->work_out, -hji, wj));
    PetscCall(MatDenseRestoreColumnVecRead(W, j, &wj));
    ws->H[(size_t)j + (size_t)i * (size_t)ldH] = hji;
  }

  PetscReal nrm;
  PetscCall(VecNorm(ws->work_out, NORM_2, &nrm));
  ws->H[(size_t)(i + 1) + (size_t)i * (size_t)ldH] = (PetscScalar)nrm;
  PetscCall(VecScale(ws->work_out, 1.0 / nrm));

  /* Write the orthonormalised vector back into W[:, i+1]. */
  {
    Vec wip1;
    PetscCall(MatDenseGetColumnVec(W, i + 1, &wip1));
    PetscCall(VecCopy(ws->work_out, wip1));
    PetscCall(MatDenseRestoreColumnVec(W, i + 1, &wip1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * gmres_m — plain Arnoldi GMRES.
 *
 * Inputs:  x_in, r_in, beta_in (with ws.m = s implicitly via the caller's
 *          workspace setup), tolabs.
 * Outputs: x_out, r_out, beta_out (= |gamma[m]| if not converged inside,
 *          else the converged-inside residual norm), and ws->{W, H, Q, R}
 *          for the cycle bodies' downstream use.
 *
 * Convention notes:
 *   - W is N x (m+1); column 0 is r_in / beta_in, columns 1..m are the
 *     successive Arnoldi vectors.
 *   - Q is (m+1) x (m+1) initialized to identity, then rotated by m
 *     Givens; the C++ port emits Q.topRows(m).transpose() (size (m+1) x m
 *     after transpose is (m+1) x m); we keep ws->Q in its full (m+1)x(m+1)
 *     form and the cycle bodies extract their slice.
 *
 * Mirrors gmstab_cpp/src/modified_gmres.cpp::gmres_m exactly.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABGmresM_Private(KSP ksp,
                                                    Vec x_in, Vec r_in, PetscReal beta_in,
                                                    PetscReal tolabs,
                                                    KSPGMSTABInnerWorkspace *ws,
                                                    Vec x_out, Vec r_out, PetscReal *beta_out,
                                                    PetscBool *converged_inside)
{
  PetscFunctionBegin;
  const PetscInt m = ws->m;
  const PetscInt ldH = m + 1;

  /* W[:, 0] = r_in / beta_in */
  Vec w0;
  PetscCall(MatDenseGetColumnVec(ws->W, 0, &w0));
  PetscCall(VecCopy(r_in, w0));
  PetscCall(VecScale(w0, 1.0 / beta_in));
  PetscCall(MatDenseRestoreColumnVec(ws->W, 0, &w0));

  /* gamma[0] = beta_in */
  ws->gamma[0] = (PetscScalar)beta_in;

  PetscBool converged = PETSC_FALSE;
  PetscReal beta = beta_in;

  for (PetscInt i = 0; i < m; ++i) {
    /* W[:, i+1] = A * W[:, i]
       Through KSP_PCApplyBAorAB so pc_side dispatch is uniform.
       PETSc MatDense only allows ONE outstanding column-vec at a time, so
       we copy W[:, i] into work_in, do the matvec into work_out (via
       KSP_PCApplyBAorAB which uses work_n internally for any PCApply),
       then copy work_out into W[:, i+1]. */
    {
      Vec wi;
      PetscCall(MatDenseGetColumnVecRead(ws->W, i, &wi));
      PetscCall(VecCopy(wi, ws->work_in));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, i, &wi));
    }
    PetscCall(KSP_PCApplyBAorAB(ksp, ws->work_in, ws->work_out, ws->work_n));
    if (ws->matvec_count_ptr) (*ws->matvec_count_ptr)++;
    {
      Vec wip1;
      PetscCall(MatDenseGetColumnVec(ws->W, i + 1, &wip1));
      PetscCall(VecCopy(ws->work_out, wip1));
      PetscCall(MatDenseRestoreColumnVec(ws->W, i + 1, &wip1));
    }

    /* Classical Gram-Schmidt + scaling. */
    PetscCall(KSPGMSTABArnoldiStep_Private(ws, i));

    /* Givens. */
    PetscCall(KSPGMSTABApplyInnerGivens_Private(ws, i));

    beta = PetscAbsScalar(ws->gamma[i + 1]);

    if (beta <= tolabs) {
      /* Build (x, r, beta) from the truncated factor: zeta = R(0..i, 0..i)^-1 * gamma(0..i). */
      PetscScalar *zeta;
      PetscCall(PetscMalloc1(i + 1, &zeta));
      PetscCall(KSPGMSTABTriUSolve_Private(ws->R, ldH, ws->gamma, zeta, i + 1));
      PetscCall(KSPGMSTABAxpyKrylov_Private(x_out, x_in, ws->W, zeta, i + 1));
      PetscCall(KSPGMSTABResidualReconstruct_Private(r_out, ws->W, ws->H, ldH, zeta, beta_in, i + 1));
      PetscCall(PetscFree(zeta));
      converged = PETSC_TRUE;
      break;
    }
  }

  if (!converged) {
    PetscCall(VecCopy(x_in, x_out));
    PetscCall(VecCopy(r_in, r_out));
    beta = beta_in;
  }

  *beta_out         = beta;
  *converged_inside = converged;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Common per-iteration "build x_tmp" used by pgmres / aug_gmres for the
 * snapshot callback. Computes:
 *   x_tmp = x + W[:, 0..i_done-1] * zeta - V0 * (Z \ (Y[:, 0..i_done-1] * zeta))
 * where zeta = R(0..i_done-1, 0..i_done-1)^-1 * gamma(0..i_done-1).
 *
 * V0 may be NULL for plain gmres_m (where the projection terms vanish);
 * Y must be NULL in that case.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABBuildXtmp_Private(KSPGMSTABInnerWorkspace *ws,
                                                  PetscInt i_done,
                                                  Vec x_in, Mat V0, const PetscScalar *Z, PetscInt Z_ldim,
                                                  Vec x_tmp_out)
{
  PetscFunctionBegin;
  const PetscInt ldH = ws->m + 1;
  PetscScalar *zeta;
  PetscCall(PetscMalloc1(i_done, &zeta));
  PetscCall(KSPGMSTABTriUSolve_Private(ws->R, ldH, ws->gamma, zeta, i_done));

  /* x_tmp = x_in + W[:, 0..i_done-1] * zeta. */
  PetscCall(KSPGMSTABAxpyKrylov_Private(x_tmp_out, x_in, ws->W, zeta, i_done));

  if (V0 && Z) {
    /* Subtract V0 * (Z \ (Y[:, 0..i_done-1] * zeta)).
       (1) rhs = Y[:, 0..i_done-1] * zeta — in host space, length s.
       (2) proj = Z \ rhs (lower-tri solve).
       (3) x_tmp -= V0 * proj_vec. */
    PetscScalar *rhs, *proj;
    PetscCall(PetscMalloc2(ws->s, &rhs, ws->s, &proj));
    for (PetscInt k = 0; k < ws->s; ++k) rhs[k] = 0.0;
    {
      PetscBLASInt M_, N_, lda_ = (PetscBLASInt)ws->s, inc = 1;
      PetscScalar  one = 1.0, zero_ = 0.0;
      PetscCall(PetscBLASIntCast(ws->s, &M_));
      PetscCall(PetscBLASIntCast(i_done, &N_));
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &M_, &N_, &one, ws->Y, &lda_, zeta, &inc, &zero_, rhs, &inc));
    }
    PetscCall(KSPGMSTABTriLSolve_Private(Z, Z_ldim, rhs, proj, ws->s));

    Vec proj_vec, scratch;
    PetscCall(VecDuplicate(x_tmp_out, &scratch));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, ws->s, proj, &proj_vec));
    /* MatMult(V0, proj_vec, scratch) only works for parallel V0 with
       parallel proj_vec; for the small s-vector we apply column-by-column
       via VecAXPY. */
    PetscCall(VecDestroy(&proj_vec));
    PetscCall(VecSet(scratch, 0.0));
    for (PetscInt k = 0; k < ws->s; ++k) {
      Vec V0k;
      PetscCall(MatDenseGetColumnVecRead(V0, k, &V0k));
      PetscCall(VecAXPY(scratch, proj[k], V0k));
      PetscCall(MatDenseRestoreColumnVecRead(V0, k, &V0k));
    }
    PetscCall(VecAXPY(x_tmp_out, -1.0, scratch));
    PetscCall(VecDestroy(&scratch));

    PetscCall(PetscFree2(rhs, proj));
  }
  PetscCall(PetscFree(zeta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * pgmres_m — projected GMRES.
 *
 * Each iteration:
 *   Y[:, i] = P^T * W[:, i]
 *   xi      = Z \ Y[:, i]                  (s × 1)
 *   v_proj  = W[:, i] - V0 * xi
 *   W[:, i+1] = A * v_proj                  (matvec via KSP_PCApplyBAorAB)
 *   ... classical Gram-Schmidt orthogonalisation against W[:, 0..i] ...
 *   ... apply Givens rotation to update R, gamma, Q ...
 *   if |gamma[i+1]| <= tolabs: build (x, r) and exit early
 *   else: snapshot via snap_cb(ksp, x_tmp, |gamma[i+1]|, ...)
 *
 * On exit (whether converged early or completed all m iterations), Y[:, m]
 * is also computed (the C++ port does this in the "if (!returned_early)"
 * tail) so the caller has Y of shape s x (m+1).
 * ============================================================================ */
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
                                                     PetscBool *terminated_early)
{
  PetscFunctionBegin;
  const PetscInt m   = ws->m;
  const PetscInt s   = ws->s;
  const PetscInt ldH = m + 1;

  /* W[:, 0] = r_in / beta_in;  gamma[0] = beta_in. */
  Vec w0;
  PetscCall(MatDenseGetColumnVec(ws->W, 0, &w0));
  PetscCall(VecCopy(r_in, w0));
  PetscCall(VecScale(w0, 1.0 / beta_in));
  PetscCall(MatDenseRestoreColumnVec(ws->W, 0, &w0));
  ws->gamma[0] = (PetscScalar)beta_in;

  PetscBool converged = PETSC_FALSE;
  PetscBool early_term = PETSC_FALSE;
  PetscReal beta = beta_in;

  /* Persistent buffers for the s-element xi vector and projected w. */
  PetscScalar *xi;
  PetscCall(PetscMalloc1(s, &xi));
  Vec proj_v;
  PetscCall(VecDuplicate(ws->work_n, &proj_v));

  for (PetscInt i = 0; i < m; ++i) {
    /* Y[:, i] = P^T * W[:, i] */
    Vec wi;
    PetscCall(MatDenseGetColumnVecRead(ws->W, i, &wi));
    PetscCall(KSPGMSTABProjectColumn_Private(P, wi, ws->Y, s, i, NULL));

    /* xi = Z \ Y[:, i] (lower-triangular). */
    PetscCall(KSPGMSTABTriLSolve_Private(Z, Z_ldim, &ws->Y[(size_t)i * (size_t)s], xi, s));

    /* proj_v = W[:, i] - V0 * xi */
    PetscCall(VecCopy(wi, proj_v));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, i, &wi));
    for (PetscInt k = 0; k < s; ++k) {
      Vec V0k;
      PetscCall(MatDenseGetColumnVecRead(V0, k, &V0k));
      PetscCall(VecAXPY(proj_v, -xi[k], V0k));
      PetscCall(MatDenseRestoreColumnVecRead(V0, k, &V0k));
    }

    /* W[:, i+1] = A * proj_v */
    Vec wip1;
    PetscCall(MatDenseGetColumnVec(ws->W, i + 1, &wip1));
    PetscCall(KSP_PCApplyBAorAB(ksp, proj_v, wip1, ws->work_n));
    if (ws->matvec_count_ptr) (*ws->matvec_count_ptr)++;
    PetscCall(MatDenseRestoreColumnVec(ws->W, i + 1, &wip1));

    /* Classical GS + Givens. */
    PetscCall(KSPGMSTABArnoldiStep_Private(ws, i));
    PetscCall(KSPGMSTABApplyInnerGivens_Private(ws, i));

    beta = PetscAbsScalar(ws->gamma[i + 1]);

    if (beta <= tolabs) {
      PetscScalar *zeta;
      PetscCall(PetscMalloc1(i + 1, &zeta));
      PetscCall(KSPGMSTABTriUSolve_Private(ws->R, ldH, ws->gamma, zeta, i + 1));

      /* x_out = x_in + W[:, 0..i] * zeta - V0 * (Z \ (Y[:, 0..i] * zeta)) */
      PetscCall(KSPGMSTABBuildXtmp_Private(ws, i + 1, x_in, V0, Z, Z_ldim, x_out));

      /* r_out = W[:, 0..i+1] * (gamma_full - H * zeta) */
      PetscCall(KSPGMSTABResidualReconstruct_Private(r_out, ws->W, ws->H, ldH, zeta, beta_in, i + 1));
      PetscCall(PetscFree(zeta));

      /* Snapshot at the converged-early state. */
      if (snap_cb) {
        Vec gp;
        PetscCall(VecDuplicate(x_out, &gp));
        PetscCall(VecCopy(x_global, gp));
        PetscCall(VecAXPY(gp, 1.0, x_out));
        PetscCall(snap_cb(ksp, gp, beta, &early_term, snap_ctx));
        PetscCall(VecDestroy(&gp));
      }
      converged = PETSC_TRUE;
      break;
    }

    /* Per-iteration snapshot. */
    if (snap_cb) {
      Vec x_tmp, gp;
      PetscCall(VecDuplicate(x_out, &x_tmp));
      PetscCall(KSPGMSTABBuildXtmp_Private(ws, i + 1, x_in, V0, Z, Z_ldim, x_tmp));
      PetscCall(VecDuplicate(x_tmp, &gp));
      PetscCall(VecCopy(x_global, gp));
      PetscCall(VecAXPY(gp, 1.0, x_tmp));
      PetscCall(snap_cb(ksp, gp, beta, &early_term, snap_ctx));
      PetscCall(VecDestroy(&gp));
      PetscCall(VecDestroy(&x_tmp));
      if (early_term) {
        /* The MATLAB / C++ port returns mid-loop. The current x stays as
           x_in; r stays as r_in (no commit yet). */
        PetscCall(VecCopy(x_in, x_out));
        PetscCall(VecCopy(r_in, r_out));
        break;
      }
    }
  }

  /* If we ran the full m iterations without converging, the C++ port
     also computes Y[:, m] (line "if (!returned_early)"). Reproduce. */
  if (!converged && !early_term) {
    Vec wm;
    PetscCall(MatDenseGetColumnVecRead(ws->W, m, &wm));
    PetscCall(KSPGMSTABProjectColumn_Private(P, wm, ws->Y, s, m, NULL));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, m, &wm));

    /* Cycle body uses (x, r, beta) as the values at the last completed
       iteration; the C++ port leaves them as x_in / r_in / beta_in
       (which matches: out.x = std::move(x); etc., where x = x_in). */
    PetscCall(VecCopy(x_in, x_out));
    PetscCall(VecCopy(r_in, r_out));
    beta = beta_in;
  }

  PetscCall(PetscFree(xi));
  PetscCall(VecDestroy(&proj_v));

  *beta_out         = beta;
  *converged_inside = converged;
  *terminated_early = early_term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * aug_gmres_m — augmented GMRES.
 *
 * Each iteration:
 *   W[:, i+1] = A * W[:, i]                 (matvec FIRST)
 *   Y[:, i]   = P^T * W[:, i+1]              (project the OUTPUT, not input)
 *   xi        = Z \ Y[:, i]
 *   W[:, i+1] -= V1 * xi                     (subtract V1, not V0)
 *   ... classical GS ...
 *   ... Givens ...
 *   if converged: build (x, r) using V0 * (Z \ (Y[:, 0..i] * zeta)) — same
 *                  formula as pgmres
 *   else: snapshot
 *
 * Note: Y here is sized s × m (one fewer column than pgmres) because
 * Y[:, i] is computed from W[:, i+1], not W[:, i]; we never evaluate
 * Y at position m.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABAugGmresM_Private(KSP ksp,
                                                       Mat P, const PetscScalar *Z, PetscInt Z_ldim,
                                                       Mat V0, Mat V1,
                                                       Vec x_in, Vec r_in, PetscReal beta_in,
                                                       PetscReal tolabs,
                                                       KSPGMSTABInnerWorkspace *ws,
                                                       Vec x_global,
                                                       KSPGMSTABInnerSnapshot snap_cb, void *snap_ctx,
                                                       Vec x_out, Vec r_out, PetscReal *beta_out,
                                                       PetscBool *converged_inside)
{
  PetscFunctionBegin;
  const PetscInt m   = ws->m;
  const PetscInt s   = ws->s;
  const PetscInt ldH = m + 1;

  /* W[:, 0] = r_in / beta_in;  gamma[0] = beta_in. */
  Vec w0;
  PetscCall(MatDenseGetColumnVec(ws->W, 0, &w0));
  PetscCall(VecCopy(r_in, w0));
  PetscCall(VecScale(w0, 1.0 / beta_in));
  PetscCall(MatDenseRestoreColumnVec(ws->W, 0, &w0));
  ws->gamma[0] = (PetscScalar)beta_in;

  PetscBool converged = PETSC_FALSE;
  PetscBool early_term = PETSC_FALSE;
  PetscReal beta = beta_in;

  PetscScalar *xi;
  PetscCall(PetscMalloc1(s, &xi));

  for (PetscInt i = 0; i < m; ++i) {
    /* W[:, i+1] = A * W[:, i] (via work_in/work_out scratch) */
    {
      Vec wi;
      PetscCall(MatDenseGetColumnVecRead(ws->W, i, &wi));
      PetscCall(VecCopy(wi, ws->work_in));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, i, &wi));
    }
    PetscCall(KSP_PCApplyBAorAB(ksp, ws->work_in, ws->work_out, ws->work_n));
    if (ws->matvec_count_ptr) (*ws->matvec_count_ptr)++;
    /* work_out now holds A*W[:,i]; project it before writing into W[:,i+1]. */

    /* Y[:, i] = P^T * work_out */
    PetscCall(KSPGMSTABProjectColumn_Private(P, ws->work_out, ws->Y, s, i, NULL));

    /* xi = Z \ Y[:, i] */
    PetscCall(KSPGMSTABTriLSolve_Private(Z, Z_ldim, &ws->Y[(size_t)i * (size_t)s], xi, s));

    /* work_out -= V1 * xi */
    for (PetscInt k = 0; k < s; ++k) {
      Vec V1k;
      PetscCall(MatDenseGetColumnVecRead(V1, k, &V1k));
      PetscCall(VecAXPY(ws->work_out, -xi[k], V1k));
      PetscCall(MatDenseRestoreColumnVecRead(V1, k, &V1k));
    }
    /* Now copy into W[:, i+1]. */
    {
      Vec wip1;
      PetscCall(MatDenseGetColumnVec(ws->W, i + 1, &wip1));
      PetscCall(VecCopy(ws->work_out, wip1));
      PetscCall(MatDenseRestoreColumnVec(ws->W, i + 1, &wip1));
    }

    /* Classical GS + Givens. */
    PetscCall(KSPGMSTABArnoldiStep_Private(ws, i));
    PetscCall(KSPGMSTABApplyInnerGivens_Private(ws, i));

    beta = PetscAbsScalar(ws->gamma[i + 1]);

    if (beta <= tolabs) {
      PetscScalar *zeta;
      PetscCall(PetscMalloc1(i + 1, &zeta));
      PetscCall(KSPGMSTABTriUSolve_Private(ws->R, ldH, ws->gamma, zeta, i + 1));

      PetscCall(KSPGMSTABBuildXtmp_Private(ws, i + 1, x_in, V0, Z, Z_ldim, x_out));
      PetscCall(KSPGMSTABResidualReconstruct_Private(r_out, ws->W, ws->H, ldH, zeta, beta_in, i + 1));
      PetscCall(PetscFree(zeta));

      if (snap_cb) {
        Vec gp;
        PetscCall(VecDuplicate(x_out, &gp));
        PetscCall(VecCopy(x_global, gp));
        PetscCall(VecAXPY(gp, 1.0, x_out));
        PetscCall(snap_cb(ksp, gp, beta, &early_term, snap_ctx));
        PetscCall(VecDestroy(&gp));
      }
      converged = PETSC_TRUE;
      break;
    }

    if (snap_cb) {
      Vec x_tmp, gp;
      PetscCall(VecDuplicate(x_out, &x_tmp));
      PetscCall(KSPGMSTABBuildXtmp_Private(ws, i + 1, x_in, V0, Z, Z_ldim, x_tmp));
      PetscCall(VecDuplicate(x_tmp, &gp));
      PetscCall(VecCopy(x_global, gp));
      PetscCall(VecAXPY(gp, 1.0, x_tmp));
      PetscCall(snap_cb(ksp, gp, beta, &early_term, snap_ctx));
      PetscCall(VecDestroy(&gp));
      PetscCall(VecDestroy(&x_tmp));
      /* aug_gmres in C++ does NOT short-circuit on early_term — there's
         no `return` mid-loop in aug_gmres_m.cpp. So we don't honour it
         here either. */
    }
  }

  if (!converged) {
    PetscCall(VecCopy(x_in, x_out));
    PetscCall(VecCopy(r_in, r_out));
    beta = beta_in;
  }

  PetscCall(PetscFree(xi));

  *beta_out         = beta;
  *converged_inside = converged;
  PetscFunctionReturn(PETSC_SUCCESS);
}
