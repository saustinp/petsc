/*
   Initialisation — direct port of gmstab_cpp/src/solver.cpp::initialisation
   (which is itself a line-by-line port of MATLAB's Initialisation.m).

   Builds the initial V0, V1, Z (lower-triangular) plus updated x, r, beta
   from a "cold start" using one inner gmres_m call followed by a sequence
   of small-dense linear-algebra updates.

   Algorithm (from C++ solver.cpp lines 46-104):

     (W, H, Q_h, R_h) := gmres_m(A, x_in, r_in, beta_in, m=s)         [s matvecs]
     Y          := P^T * W                              (s × (s+1))
     eta        := beta_in * Y(:, 0)                    (s × 1)
     Z          := Y * Q_h                              (s × s)
     [Q_z, L_z] := lq(Z)                                Z = L_z * Q_z'
     xi         := R_h \ (Q_z * (L_z \ eta))            (s × 1)
     gamma_vec  := [beta_in; 0; …; 0]                   (s+1 × 1)
     c0         := gamma_vec - H * xi                   (s+1 × 1)
     Z          := L_z   (lower-triangular)
     x_out      := x_in + W(:, 0..s-1) * xi
     r_out      := W(:, 0..s)        * c0
     beta_out   := ||c0||_2
     V0         := W(:, 0..s-1) * (R_h \ Q_z)
     V1         := W(:, 0..s)   * (Q_h * Q_z)

   Sizes (with s=4):
     W is N × 5.
     H is 5 × 4.
     Q_h is 5 × 4 (the C++ port stores Q.topRows(m).transpose() — see
       gmres_m output convention).
     R_h is 4 × 4.
     Y is s × (m+1) = 4 × 5.
     Z, L_z, Q_z all 4 × 4.
*/
#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_internal.h>
#include <math.h>
#include <string.h>

PETSC_INTERN PetscErrorCode KSPGMSTABInitialisation_Private(KSP ksp, KSP_GMSTAB *gms,
                                                             KSPGMSTABInnerWorkspace *ws,
                                                             Vec x_local, Vec r, PetscReal *beta_io)
{
  PetscFunctionBegin;
  const PetscInt s = gms->s;

  /* (1) Inner gmres_m with m = s. */
  ws->m = s;
  /* The inner workspace is reset before each inner-solve to clean state. */
  {
    /* manual reset of ws->H/R/Q/G/gamma/Y for THIS inner call */
    const PetscInt mp1 = s + 1;
    PetscCall(PetscArrayzero(ws->H, (size_t)mp1 * (size_t)s));
    PetscCall(PetscArrayzero(ws->R, (size_t)mp1 * (size_t)s));
    PetscCall(PetscArrayzero(ws->Q, (size_t)mp1 * (size_t)mp1));
    PetscCall(PetscArrayzero(ws->Y, (size_t)s   * (size_t)mp1));
    PetscCall(PetscArrayzero(ws->gamma, (size_t)mp1));
    PetscCall(PetscArrayzero(ws->G, (size_t)s));
    /* Q := identity. */
    for (PetscInt i = 0; i < mp1; ++i) ws->Q[(size_t)i + (size_t)i * (size_t)mp1] = 1.0;
  }

  PetscReal beta_in = *beta_io;
  Vec  x_init, r_init;
  PetscCall(VecDuplicate(x_local, &x_init));
  PetscCall(VecDuplicate(r,       &r_init));
  PetscCall(VecCopy(x_local, x_init));
  PetscCall(VecCopy(r,       r_init));

  PetscReal beta_g;
  PetscBool gmres_converged;
  PetscCall(KSPGMSTABGmresM_Private(ksp, x_init, r_init, beta_in, /*tolabs*/ ksp->abstol,
                                    ws, x_local, r, &beta_g, &gmres_converged));

  /* If the inner GMRES converged before s steps the C++ port returns from
     Initialisation without populating V0/V1/Z, leaving the caller to see
     it via beta. Mirror that. */
  if (gmres_converged) {
    *beta_io = beta_g;
    PetscCall(VecDestroy(&x_init));
    PetscCall(VecDestroy(&r_init));
    PetscCall(KSPGMSTABSnapshot_Private(ksp, gms, x_local, beta_g));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* (2) Y := P^T * W, all s+1 columns.
     ws->Y is the s × (s+1) buffer. ws->W is N × (s+1) MatDense. */
  for (PetscInt j = 0; j <= s; ++j) {
    Vec wj;
    PetscCall(MatDenseGetColumnVecRead(ws->W, j, &wj));
    for (PetscInt k = 0; k < s; ++k) {
      Vec pk;
      PetscScalar yk;
      PetscCall(MatDenseGetColumnVecRead(gms->P, k, &pk));
      PetscCall(VecDot(wj, pk, &yk));
      PetscCall(MatDenseRestoreColumnVecRead(gms->P, k, &pk));
      ws->Y[(size_t)k + (size_t)j * (size_t)s] = yk;
    }
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, j, &wj));
  }

  /* (3) eta = beta_in * Y(:, 0). */
  PetscScalar *eta;
  PetscCall(PetscMalloc1(s, &eta));
  for (PetscInt k = 0; k < s; ++k) eta[k] = (PetscScalar)beta_in * ws->Y[(size_t)k + 0];

  /* (4) Q_h is in ws->Q. The C++ port emits `Q = Q.topRows(m).transpose()`,
         meaning the output Q has shape (s+1) × s in column-major terms.
         Looking at the Q layout we maintain in the inner GMRES: ws->Q is
         the FULL (s+1) × (s+1) accumulated Givens product. The "Qh"
         used here is ws->Q.topRows(s).transpose() = the FIRST s rows of
         ws->Q, but TRANSPOSED. After transpose, that's an (s+1) × s
         matrix in column-major.

         Concretely:  Qh[a, b] = ws->Q[b, a]    with a ∈ [0, s+1), b ∈ [0, s).

         R_h is ws->R(0:s, 0:s) — the upper-triangular rotated factor.
  */
  const PetscInt mp1 = s + 1;
  PetscScalar *Qh;
  PetscCall(PetscMalloc1((size_t)mp1 * (size_t)s, &Qh));
  for (PetscInt b = 0; b < s; ++b)
    for (PetscInt a = 0; a < mp1; ++a)
      Qh[(size_t)a + (size_t)b * (size_t)mp1] = ws->Q[(size_t)b + (size_t)a * (size_t)mp1];

  PetscScalar *Rh;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &Rh));
  for (PetscInt b = 0; b < s; ++b)
    for (PetscInt a = 0; a <= b; ++a)
      Rh[(size_t)a + (size_t)b * (size_t)s] = ws->R[(size_t)a + (size_t)b * (size_t)mp1];
  /* Zero out the strict lower triangle of Rh. */
  for (PetscInt b = 0; b < s; ++b)
    for (PetscInt a = b + 1; a < s; ++a)
      Rh[(size_t)a + (size_t)b * (size_t)s] = 0.0;

  /* (5) Z := Y * Q_h         (s × s)
         Y is s × (s+1) col-major (leading dim s).
         Q_h is (s+1) × s      col-major (leading dim s+1).
         Result Z is s × s     col-major (leading dim s). */
  PetscScalar *Z_buf;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &Z_buf));
  PetscCall(KSPGMSTABGemm_Private("N", "N", s, s, mp1,
            1.0, ws->Y, s, Qh, mp1,
            0.0, Z_buf, s));

  /* (6) [Q_z, L_z] = lq(Z_buf)  — Q_z is s×s, L_z is s×s lower-triangular. */
  PetscScalar *Qz, *Lz;
  PetscCall(PetscMalloc2((size_t)s * (size_t)s, &Qz, (size_t)s * (size_t)s, &Lz));
  PetscCall(KSPGMSTABLq_Private(Z_buf, s, s, s, Qz, s, Lz, s));

  /* (7) xi = Rh \ (Qz * (Lz \ eta)).
         Step 7a: solve L_z * inner = eta  (lower-tri).
         Step 7b: temp = Q_z * inner.
         Step 7c: solve R_h * xi = temp    (upper-tri). */
  PetscScalar *inner, *xi, *temp;
  PetscCall(PetscMalloc3(s, &inner, s, &xi, s, &temp));
  for (PetscInt k = 0; k < s; ++k) inner[k] = eta[k];
  PetscCall(KSPGMSTABTrsv_Private("L", "N", "N", s, Lz, s, inner, 1));
  PetscCall(KSPGMSTABGemv_Private("N", s, s, 1.0, Qz, s, inner, 1, 0.0, temp, 1));
  for (PetscInt k = 0; k < s; ++k) xi[k] = temp[k];
  PetscCall(KSPGMSTABTrsv_Private("U", "N", "N", s, Rh, s, xi, 1));

  /* (8) gamma_vec = [beta_in; 0; …]; c0 = gamma_vec - H * xi. */
  PetscScalar *c0;
  PetscCall(PetscMalloc1(mp1, &c0));
  c0[0] = (PetscScalar)beta_in;
  for (PetscInt k = 1; k < mp1; ++k) c0[k] = 0.0;
  PetscCall(KSPGMSTABGemv_Private("N", mp1, s, -1.0, ws->H, mp1, xi, 1, 1.0, c0, 1));

  /* (9) x_out = x_in + W(:, 0..s-1) * xi. (We update the in-place x_local.) */
  for (PetscInt k = 0; k < s; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
    PetscCall(VecAXPY(x_local, xi[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
  }
  /* Reset r_local before accumulating from c0. */
  PetscCall(VecSet(r, 0.0));
  for (PetscInt k = 0; k <= s; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
    PetscCall(VecAXPY(r, c0[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
  }

  /* beta = ||c0||_2 (the post-Initialisation residual norm). */
  PetscReal cn = 0.0;
  for (PetscInt k = 0; k <= s; ++k) cn += PetscRealPart(c0[k]) * PetscRealPart(c0[k]);
  PetscReal beta_out = PetscSqrtReal(cn);

  /* (10) Z := L_z (lower-triangular). Persist into gms->Z. */
  if (!gms->Z) {
    PetscCall(PetscMalloc1((size_t)s * (size_t)s, &gms->Z));
    gms->Z_ldim = s;
  }
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < s; ++rr)
      gms->Z[(size_t)rr + (size_t)c_ * (size_t)s] = Lz[(size_t)rr + (size_t)c_ * (size_t)s];

  /* (11) V0 := W(:, 0..s-1) * (R_h \ Q_z).
         R_h \ Q_z = upper-tri-solve, in place: copy Q_z into a buffer, then
         BLAS trsm with side='L', uplo='U'. Result is s × s.
         Then V0 (N × s MatDense) gets each column k as
         sum_l W(:, l) * (R_h \ Q_z)(l, k). */
  PetscScalar *RhQz;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &RhQz));
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < s; ++rr)
      RhQz[(size_t)rr + (size_t)c_ * (size_t)s] = Qz[(size_t)rr + (size_t)c_ * (size_t)s];
  PetscCall(KSPGMSTABTrsm_Private("L", "U", "N", "N", s, s, 1.0, Rh, s, RhQz, s));

  if (!gms->V0) {
    PetscInt n_local;
    PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &gms->V0));
    PetscCall(MatSetType(gms->V0, MATDENSE));
    PetscCall(MatSetSizes(gms->V0, n_local, PETSC_DECIDE, N_global, s));
    PetscCall(MatSetUp(gms->V0));
  }
  for (PetscInt c_ = 0; c_ < s; ++c_) {
    Vec V0c;
    PetscCall(MatDenseGetColumnVec(gms->V0, c_, &V0c));
    PetscCall(VecSet(V0c, 0.0));
    for (PetscInt k = 0; k < s; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V0c, RhQz[(size_t)k + (size_t)c_ * (size_t)s], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    PetscCall(MatDenseRestoreColumnVec(gms->V0, c_, &V0c));
  }

  /* (12) V1 := W(:, 0..s) * (Q_h * Q_z).   QhQz is (s+1) × s. */
  PetscScalar *QhQz;
  PetscCall(PetscMalloc1((size_t)mp1 * (size_t)s, &QhQz));
  PetscCall(KSPGMSTABGemm_Private("N", "N", mp1, s, s, 1.0, Qh, mp1, Qz, s, 0.0, QhQz, mp1));

  if (!gms->V1) {
    PetscInt n_local;
    PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &gms->V1));
    PetscCall(MatSetType(gms->V1, MATDENSE));
    PetscCall(MatSetSizes(gms->V1, n_local, PETSC_DECIDE, N_global, s));
    PetscCall(MatSetUp(gms->V1));
  }
  for (PetscInt c_ = 0; c_ < s; ++c_) {
    Vec V1c;
    PetscCall(MatDenseGetColumnVec(gms->V1, c_, &V1c));
    PetscCall(VecSet(V1c, 0.0));
    for (PetscInt k = 0; k <= s; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V1c, QhQz[(size_t)k + (size_t)c_ * (size_t)mp1], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    PetscCall(MatDenseRestoreColumnVec(gms->V1, c_, &V1c));
  }

  *beta_io = beta_out;

  /* Snapshot per the C++ port (Initialisation calls perf.read at end). */
  PetscCall(KSPGMSTABSnapshot_Private(ksp, gms, x_local, beta_out));

  PetscCall(PetscFree(c0));
  PetscCall(PetscFree3(inner, xi, temp));
  PetscCall(PetscFree2(Qz, Lz));
  PetscCall(PetscFree(Z_buf));
  PetscCall(PetscFree(Rh));
  PetscCall(PetscFree(Qh));
  PetscCall(PetscFree(eta));
  PetscCall(PetscFree(RhQz));
  PetscCall(PetscFree(QhQz));
  PetscCall(VecDestroy(&x_init));
  PetscCall(VecDestroy(&r_init));
  PetscFunctionReturn(PETSC_SUCCESS);
}
