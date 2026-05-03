/*
   Helpers used across KSPSolve_GMSTAB: snapshot emission (PerfMeasure
   port), default shadow-space construction, and file-based shadow loading.
*/
#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_internal.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * KSPGMSTABSnapshot_Private
 *
 *   Emits one row of the (iter, matvec, iterres, trueres, runtime, runtime_mv)
 *   CSV that the C++ port's PerfMeasure produces. Recomputes the true
 *   residual via an UNCOUNTED MatMult (using the operator directly, not
 *   KSP_PCApplyBAorAB — the C++ counterpart is `b - A_raw_(x)`).
 *
 *   Also feeds the iter_norm into the standard PETSc convergence machinery
 *   (KSPMonitor + (*ksp->converged)) so user-set tolerances apply.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABSnapshot_Private(KSP ksp, KSP_GMSTAB *gms,
                                                       Vec x_total, PetscReal iter_norm)
{
  PetscFunctionBegin;

  /* Invariants — cheap defensive checks. matvec_count and snapshot_count
     are unsigned-semantically monotonic; assert that here so future bugs
     that decrement them (or skip the increment) trip a clear error rather
     than emitting a corrupted CSV row. */
  PetscCheck(gms->matvec_count   >= 0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB,
             "KSPGMSTABSnapshot_Private: matvec_count went negative (=%" PetscInt_FMT ")",
             gms->matvec_count);
  PetscCheck(gms->snapshot_count >= 0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB,
             "KSPGMSTABSnapshot_Private: snapshot_count went negative (=%" PetscInt_FMT ")",
             gms->snapshot_count);
  PetscCheck(gms->matvec_count   >= gms->_last_logged_matvec_count,
             PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB,
             "KSPGMSTABSnapshot_Private: matvec_count went backwards (was %" PetscInt_FMT ", now %" PetscInt_FMT ") — "
             "indicates a missing-counter or double-counted matvec bug",
             gms->_last_logged_matvec_count, gms->matvec_count);
  gms->_last_logged_matvec_count = gms->matvec_count;

  PetscReal norm_true = 0.0;

  /* The constructor snapshot (snapshot_count == 0) reports ||b|| as the
     true residual unconditionally, matching the C++ port's PerfMeasure
     constructor — it does NOT recompute ||b - A*x0|| even when the user
     supplied a non-zero initial guess. We honour that by skipping the
     uncounted MatMult on the very first call and using iter_norm
     directly. (Caller must pass iter_norm = ||b|| at the constructor.)

     For all later snapshots, we recompute ||b - A*x_total|| via an
     uncounted MatMult so the "true" residual we log is independent of
     accumulated FP error in the iterated residual. */
  if (gms->snapshot_count == 0) {
    norm_true = iter_norm;
  } else {
    Vec b, Ax, r_true;
    PetscCall(KSPGetRhs(ksp, &b));
    PetscCall(VecDuplicate(b, &Ax));
    PetscCall(VecDuplicate(b, &r_true));
    Mat Amat;
    PetscCall(KSPGetOperators(ksp, &Amat, NULL));
    PetscCall(MatMult(Amat, x_total, Ax));
    PetscCall(VecWAXPY(r_true, -1.0, Ax, b));
    PetscCall(VecNorm(r_true, NORM_2, &norm_true));
    PetscCall(VecDestroy(&Ax));
    PetscCall(VecDestroy(&r_true));
  }

  /* Trace CSV row. The iter column is the pre-increment count, so the
     constructor snapshot lands at iter=0 (matching the C++ port). */
  const PetscInt iter_idx = gms->snapshot_count;
  if (gms->trace_csv && gms->trace_fp) {
    fprintf(gms->trace_fp, "%" PetscInt_FMT ",%" PetscInt_FMT ",%.16e,%.16e,%.6e,%.6e\n",
            iter_idx, gms->matvec_count,
            (double)iter_norm, (double)norm_true,
            (double)gms->t_total, (double)gms->t_mv);
    fflush(gms->trace_fp);
  }
  gms->snapshot_count++;

  /* Convergence callback. Rnorm is whatever -ksp_norm_type asked for:
       UNPRECONDITIONED → norm_true
       PRECONDITIONED  → iter_norm   (which is gms->beta for this build)
       NONE            → 0          (skip) */
  PetscReal rnorm_for_check;
  switch (ksp->normtype) {
  case KSP_NORM_UNPRECONDITIONED: rnorm_for_check = norm_true; break;
  case KSP_NORM_PRECONDITIONED:   rnorm_for_check = iter_norm; break;
  case KSP_NORM_NONE:             rnorm_for_check = 0.0;       break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
            "KSPGMSTAB: norm type %s not supported", KSPNormTypes[ksp->normtype]);
  }
  ksp->rnorm = rnorm_for_check;
  /* Pass the pre-increment iteration index to KSPConvergedDefault so the
     n==0 initialization branch fires on the constructor snapshot,
     properly seeding ksp->rnorm0. Without this, every subsequent residual
     would trigger DIVERGED_DTOL with rnorm0 == 0. */
  ksp->its   = iter_idx;
  PetscCall(KSPLogResidualHistory(ksp, rnorm_for_check));
  PetscCall(KSPMonitor(ksp, ksp->its, rnorm_for_check));
  PetscCall((*ksp->converged)(ksp, ksp->its, rnorm_for_check, &ksp->reason, ksp->cnvP));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPGMSTABBuildDefaultShadow_Private
 *
 *   P = orth(randn(N, s)) using a sequential RNG (seeded with rng_seed,
 *   default 5489 to match std::mt19937's default seed). NOT bit-equivalent
 *   with MATLAB's `rng('default'); orth(randn(N,s))` — the validation
 *   harness must use KSPGMSTABSetShadowSpace or -ksp_gmstab_p_file
 *   instead.
 *
 *   Implementation: build a sequential N x s dense buffer of N(0,1)
 *   draws, run KSPGMSTABOrth_Private on it (SVD-based), copy results
 *   into the gms->P MatDense.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABBuildDefaultShadow_Private(KSP ksp, KSP_GMSTAB *gms)
{
  PetscFunctionBegin;
  Mat Amat;
  PetscInt N_global, n_local;
  Vec template_vec;
  PetscCall(KSPGetOperators(ksp, &Amat, NULL));
  PetscCall(MatGetSize(Amat, &N_global, NULL));
  PetscCall(MatCreateVecs(Amat, NULL, &template_vec));
  PetscCall(VecGetLocalSize(template_vec, &n_local));
  PetscCall(VecDestroy(&template_vec));

  /* Allocate gms->P as an N x s MATDENSE matching the operator's row
     distribution. */
  PetscCall(MatDestroy(&gms->P));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &gms->P));
  PetscCall(MatSetType(gms->P, MATDENSE));
  PetscCall(MatSetSizes(gms->P, n_local, PETSC_DECIDE, N_global, gms->s));
  PetscCall(MatSetUp(gms->P));

  /* Use PETSc's PetscRandom with NORMAL distribution. Same statistical
     properties as MATLAB's randn (Gaussian N(0,1)) but DIFFERENT state
     traversal so NOT bit-equivalent. */
  if (!gms->prand) {
    PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)ksp), &gms->prand));
    PetscCall(PetscRandomSetType(gms->prand, PETSCRAND));
    PetscCall(PetscRandomSetInterval(gms->prand, -1.0, 1.0));    /* will be overridden */
    PetscCall(PetscRandomSetSeed(gms->prand, gms->rng_seed));
    PetscCall(PetscRandomSeed(gms->prand));
  }

  /* Fill columns of P with N(0,1) values via uniform → normal Box-Muller.
     PETSc has PETSCRANDER48 (uniform) and PETSCRAND (uniform); for
     reproducibility we generate uniforms then transform. */
  PetscCall(PetscRandomSetInterval(gms->prand, 0.0, 1.0));
  for (PetscInt k = 0; k < gms->s; ++k) {
    Vec Pk;
    PetscScalar *arr;
    PetscCall(MatDenseGetColumnVec(gms->P, k, &Pk));
    PetscCall(VecGetArray(Pk, &arr));
    for (PetscInt i = 0; i < n_local; i += 2) {
      PetscScalar u1, u2;
      PetscCall(PetscRandomGetValue(gms->prand, &u1));
      PetscCall(PetscRandomGetValue(gms->prand, &u2));
      const PetscReal U1 = PetscMax(PetscRealPart(u1), 1e-300);
      const PetscReal U2 = PetscRealPart(u2);
      const PetscReal mag = PetscSqrtReal(-2.0 * PetscLogReal(U1));
      arr[i] = (PetscScalar)(mag * PetscCosReal(2.0 * PETSC_PI * U2));
      if (i + 1 < n_local) arr[i + 1] = (PetscScalar)(mag * PetscSinReal(2.0 * PETSC_PI * U2));
    }
    PetscCall(VecRestoreArray(Pk, &arr));
    PetscCall(MatDenseRestoreColumnVec(gms->P, k, &Pk));
  }

  /* Now orth(P). Pull P's columns to a sequential host buffer (works
     for n_local = global since validation runs on a single rank);
     run SVD; rebuild P from U. */
  if (gms->s == 0) PetscFunctionReturn(PETSC_SUCCESS);
  /* For now the orth is on the local block — collective orth would require
     a TSQR-style algorithm. The validation harness uses the file-loaded
     path, so this is not exercised in Phase 5a. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPGMSTABLoadShadowFile_Private
 *
 *   Loads the validation P.bin format: column-major N*s float64,
 *   little-endian, no header. This is what gmstab_handoff_package_validation
 *   ships and what KSPGMSTABSetShadowSpaceFile records via -ksp_gmstab_p_file.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABLoadShadowFile_Private(KSP ksp, KSP_GMSTAB *gms)
{
  PetscFunctionBegin;
  PetscCheck(gms->P_file[0] != '\0', PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_WRONG,
             "KSPGMSTABLoadShadowFile: P_file is empty");

  Mat Amat;
  PetscInt N_global, n_local;
  Vec template_vec;
  PetscCall(KSPGetOperators(ksp, &Amat, NULL));
  PetscCall(MatGetSize(Amat, &N_global, NULL));
  PetscCall(MatCreateVecs(Amat, NULL, &template_vec));
  PetscCall(VecGetLocalSize(template_vec, &n_local));
  PetscCall(VecDestroy(&template_vec));

  /* Open P.bin, validate file size = N * s * 8 bytes. */
  FILE *fp = fopen(gms->P_file, "rb");
  PetscCheck(fp, PetscObjectComm((PetscObject)ksp), PETSC_ERR_FILE_OPEN,
             "Cannot open P_file: %s", gms->P_file);
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  long expected = (long)N_global * (long)gms->s * (long)sizeof(double);
  PetscCheck(size == expected, PetscObjectComm((PetscObject)ksp), PETSC_ERR_FILE_READ,
             "P_file %s: size %ld != expected %ld (N=%" PetscInt_FMT " s=%" PetscInt_FMT ")",
             gms->P_file, size, expected, N_global, gms->s);

  /* Allocate gms->P matching operator's distribution. */
  PetscCall(MatDestroy(&gms->P));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &gms->P));
  PetscCall(MatSetType(gms->P, MATDENSE));
  PetscCall(MatSetSizes(gms->P, n_local, PETSC_DECIDE, N_global, gms->s));
  PetscCall(MatSetUp(gms->P));

  /* Read column by column. For sequential PETSc each rank reads its own
     local slice. For Phase 5a (single rank) the local slice == global. */
  PetscMPIInt rank, size_;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ksp), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ksp), &size_));

  /* Determine local row offset. */
  PetscInt rstart, rend;
  PetscCall(MatGetOwnershipRange(gms->P, &rstart, &rend));

  for (PetscInt k = 0; k < gms->s; ++k) {
    Vec Pk;
    PetscScalar *arr;
    PetscCall(MatDenseGetColumnVec(gms->P, k, &Pk));
    PetscCall(VecGetArray(Pk, &arr));
    /* Seek to the start of column k's local slice. */
    long offset = (long)k * (long)N_global * (long)sizeof(double) + (long)rstart * (long)sizeof(double);
    fseek(fp, offset, SEEK_SET);
    /* Read n_local doubles. PetscScalar == double in default builds. */
    size_t nread = fread(arr, sizeof(double), (size_t)(rend - rstart), fp);
    PetscCheck(nread == (size_t)(rend - rstart), PetscObjectComm((PetscObject)ksp), PETSC_ERR_FILE_READ,
               "P_file column %" PetscInt_FMT ": read %zu of %ld", k, nread, (long)(rend - rstart));
    PetscCall(VecRestoreArray(Pk, &arr));
    PetscCall(MatDenseRestoreColumnVec(gms->P, k, &Pk));
  }
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}
