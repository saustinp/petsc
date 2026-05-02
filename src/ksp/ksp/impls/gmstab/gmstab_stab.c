/*
   StabCoeffs — polynomial stabilization for the IDR(s)stab core update.

   Direct port of gmstab_cpp/src/stab_coeffs.cpp. The 35° angle threshold is
   the maintaining-the-convergence guarantee from the original GM(s)stab
   paper; the rescue-on-negative-beta_sq branch matches MATLAB's behaviour
   exactly.
*/
#include <petsc/private/petscimpl.h>
#include <petscvec.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_stab.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_smalldense.h>
#include <math.h>

/* Helper: Gram matrix S(i,j) = r_i^T * r_j over a length-(L+1) bundle.
   Output S is column-major, leading dim L+1. */
static PetscErrorCode build_gram(Vec *r, PetscInt L, PetscScalar *S, PetscInt ldS)
{
  PetscFunctionBegin;
  const PetscInt Lp1 = L + 1;
  for (PetscInt i = 0; i < Lp1; ++i)
    for (PetscInt j = 0; j < Lp1; ++j)
      S[(size_t)i + (size_t)j * (size_t)ldS] = 0.0;

  /* Upper-triangle fill including diagonal: S(j, i) for j ≤ i. */
  for (PetscInt i = 0; i < Lp1; ++i) {
    for (PetscInt j = 0; j <= i; ++j) {
      PetscScalar dot;
      PetscCall(VecDot(r[i], r[j], &dot));
      S[(size_t)j + (size_t)i * (size_t)ldS] = dot;
    }
  }
  /* Symmetrize (mirror upper to lower). */
  for (PetscInt i = 0; i < Lp1; ++i)
    for (PetscInt j = 0; j < i; ++j)
      S[(size_t)i + (size_t)j * (size_t)ldS] = S[(size_t)j + (size_t)i * (size_t)ldS];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABStabCoeffs_Private(Vec *r, PetscInt L,
                                                         PetscReal beta_in, PetscReal alpha,
                                                         PetscScalar *tau, PetscReal *beta_new)
{
  PetscFunctionBegin;
  PetscCheck(L >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "L must be >= 1");
  const PetscInt Lp1 = L + 1;

  PetscScalar *S;
  PetscCall(PetscMalloc1((size_t)Lp1 * (size_t)Lp1, &S));
  PetscCall(build_gram(r, L, S, Lp1));
  /* Override S(0,0) with beta^2 (matches MATLAB). */
  S[0] = (PetscScalar)(beta_in * beta_in);
  /* Re-symmetrize since S(0,j) for j>0 was filled from a dot, S(j,0)
     was mirrored; setting S(0,0) doesn't break symmetry. */

  /* y0, yL : length Lp1, both zero except for the unit endpoints. */
  PetscScalar *y0, *yL;
  PetscCall(PetscMalloc2(Lp1, &y0, Lp1, &yL));
  for (PetscInt i = 0; i < Lp1; ++i) { y0[i] = 0.0; yL[i] = 0.0; }
  y0[0]      = 1.0;
  yL[Lp1-1]  = 1.0;

  if (L >= 2) {
    /* S_inner = S(1:L-1, 1:L-1)   (size L-1 x L-1)
       rhs0    = -S(1:L-1, 0)
       rhsL    = -S(1:L-1, L)
       Solve in-place (general LU), pack into y0[1..L-1] and yL[1..L-1].

       Build S_inner_buf as a fresh column-major array (gesv destroys A). */
    const PetscInt im = L - 1;
    PetscScalar *Si;
    PetscCall(PetscMalloc1((size_t)im * (size_t)im, &Si));
    for (PetscInt c = 0; c < im; ++c)
      for (PetscInt rr = 0; rr < im; ++rr)
        Si[(size_t)rr + (size_t)c * (size_t)im] = S[(size_t)(1 + rr) + (size_t)(1 + c) * (size_t)Lp1];

    PetscScalar *B;
    PetscCall(PetscMalloc1((size_t)im * 2, &B));   /* two RHS columns */
    for (PetscInt rr = 0; rr < im; ++rr) {
      B[(size_t)rr + (size_t)0 * (size_t)im] = -S[(size_t)(1 + rr) + (size_t)0       * (size_t)Lp1];
      B[(size_t)rr + (size_t)1 * (size_t)im] = -S[(size_t)(1 + rr) + (size_t)(Lp1-1) * (size_t)Lp1];
    }
    PetscCall(KSPGMSTABGesv_Private(im, 2, Si, im, B, im));
    for (PetscInt rr = 0; rr < im; ++rr) {
      y0[1 + rr] = B[(size_t)rr + (size_t)0 * (size_t)im];
      yL[1 + rr] = B[(size_t)rr + (size_t)1 * (size_t)im];
    }
    PetscCall(PetscFree(B));
    PetscCall(PetscFree(Si));
  }

  /* kappa0 = sqrt(y0' * S * y0); same for kappaL.
     rho     = (yL' * S * y0) / (kappa0 * kappaL). */
  PetscScalar *Sy0, *SyL;
  PetscCall(PetscMalloc2(Lp1, &Sy0, Lp1, &SyL));
  PetscCall(KSPGMSTABGemv_Private("N", Lp1, Lp1, 1.0, S, Lp1, y0, 1, 0.0, Sy0, 1));
  PetscCall(KSPGMSTABGemv_Private("N", Lp1, Lp1, 1.0, S, Lp1, yL, 1, 0.0, SyL, 1));

  PetscScalar y0Sy0 = 0.0, yLSyL = 0.0, yLSy0 = 0.0;
  for (PetscInt i = 0; i < Lp1; ++i) {
    y0Sy0 += y0[i] * Sy0[i];
    yLSyL += yL[i] * SyL[i];
    yLSy0 += yL[i] * Sy0[i];
  }
  PetscReal kappa0 = PetscSqrtReal(PetscRealPart(y0Sy0));
  PetscReal kappaL = PetscSqrtReal(PetscRealPart(yLSyL));
  PetscReal rho    = PetscRealPart(yLSy0) / (kappa0 * kappaL);

  PetscReal delta;
  if (PetscAbsReal(rho) > PetscSinReal(alpha)) {
    delta = -kappa0 / kappaL * rho;
  } else {
    /* Forced-angle (maintaining-convergence) branch. */
    PetscReal sgn = (rho > 0.0) ? 1.0 : ((rho < 0.0) ? -1.0 : 0.0);
    PetscReal denom = alpha + (PETSC_PI / 4.0) * (sgn + 1.0) - PetscAcosReal(rho);
    delta = -kappa0 / kappaL * PetscSinReal(alpha) / denom;
  }

  /* y0_new = y0 + delta * yL */
  PetscScalar *y0_new;
  PetscCall(PetscMalloc1(Lp1, &y0_new));
  for (PetscInt i = 0; i < Lp1; ++i) y0_new[i] = y0[i] + (PetscScalar)delta * yL[i];

  /* tau(0..L-1) = -y0_new(1..L). */
  for (PetscInt i = 0; i < L; ++i) tau[i] = -y0_new[1 + i];

  /* beta_sq = y0_new' * S * y0_new. */
  PetscScalar *Sy0n;
  PetscCall(PetscMalloc1(Lp1, &Sy0n));
  PetscCall(KSPGMSTABGemv_Private("N", Lp1, Lp1, 1.0, S, Lp1, y0_new, 1, 0.0, Sy0n, 1));
  PetscScalar beta_sq = 0.0;
  for (PetscInt i = 0; i < Lp1; ++i) beta_sq += y0_new[i] * Sy0n[i];

  if (PetscRealPart(beta_sq) < 0.0) {
    /* Rescue: restore S(0,0) from <r0, r0> and recompute. */
    PetscScalar r0_dot;
    PetscCall(VecDot(r[0], r[0], &r0_dot));
    S[0] = r0_dot;
    /* recompute Sy0n */
    PetscCall(KSPGMSTABGemv_Private("N", Lp1, Lp1, 1.0, S, Lp1, y0_new, 1, 0.0, Sy0n, 1));
    beta_sq = 0.0;
    for (PetscInt i = 0; i < Lp1; ++i) beta_sq += y0_new[i] * Sy0n[i];
    if (PetscRealPart(beta_sq) < 0.0) beta_sq = 0.0;
  }
  *beta_new = PetscSqrtReal(PetscRealPart(beta_sq));

  PetscCall(PetscFree(Sy0n));
  PetscCall(PetscFree(y0_new));
  PetscCall(PetscFree2(Sy0, SyL));
  PetscCall(PetscFree2(y0, yL));
  PetscCall(PetscFree(S));
  PetscFunctionReturn(PETSC_SUCCESS);
}
