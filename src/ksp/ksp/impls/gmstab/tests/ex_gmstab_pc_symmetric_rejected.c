/*
   Phase 4 final-audit tripwire — verify PC_SYMMETRIC is rejected.

   PC_SYMMETRIC was registered as supported in Phase 4a (priorities 2 and 1
   for NORM_PRECONDITIONED and NORM_UNPRECONDITIONED) but the helper code
   at gmstab_helpers.c:FinalizeSolution_Private and Snapshot_Private treated
   it as a synonym for PC_RIGHT — using full PCApply for the unwrap rather
   than PCApplySymmetricRight that split-symmetric preconditioners need.
   Silently wrong-answer for any user setting PC_SYMMETRIC.

   Phase 4 final audit dropped PC_SYMMETRIC from KSPSetSupportedNorm so
   PETSc setup rejects the side at solve time, AND added a defense-in-depth
   PetscCheck at the top of KSPSolve_GMSTAB that errors with PETSC_ERR_SUP
   in case any path bypasses the registration check (e.g., a future
   refactor re-adds the registration without fixing the helpers).

   This tripwire locks in the contract: KSPSolve with PC_SYMMETRIC must
   error, NOT silently produce a wrong x. PASS if the solve errors;
   FAIL if it returns success with rnorm != ext_res or if the runtime
   crashes / hangs.

   When Phase 4c implements proper PC_SYMMETRIC support, this tripwire
   should be REPLACED with a positive correctness test, not just deleted.
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINSYS  "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"

static PetscErrorCode load_linsys_parallel(MPI_Comm comm, const char *path, Mat *A_out, Vec *b_out)
{
  FILE *fp = fopen(path, "rb");
  PetscFunctionBegin;
  PetscCheck(fp, comm, PETSC_ERR_FILE_OPEN, "Cannot open %s", path);
  char magic[8];
  fread(magic, 1, 8, fp);
  PetscCheck(memcmp(magic, "EXASIMLS", 8) == 0, comm, PETSC_ERR_FILE_READ, "magic");
  int version, idx_size, val_size, reserved;
  fread(&version, sizeof(int), 1, fp); fread(&idx_size, sizeof(int), 1, fp);
  fread(&val_size, sizeof(int), 1, fp); fread(&reserved, sizeof(int), 1, fp);
  unsigned long long N_u, nnz_u;
  fread(&N_u, 8, 1, fp); fread(&nnz_u, 8, 1, fp);
  fseek(fp, 64, SEEK_SET);
  const PetscInt N   = (PetscInt)N_u;
  const PetscInt nnz = (PetscInt)nnz_u;
  PetscCheck(idx_size == 4, comm, PETSC_ERR_FILE_READ, "idx_size != 4");
  int *ai = (int *)malloc(sizeof(int) * (N + 1));
  int *aj = (int *)malloc(sizeof(int) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *b_arr = (double *)malloc(sizeof(double) * N);
  fread(ai, sizeof(int), N + 1, fp); fread(aj, sizeof(int), nnz, fp);
  fread(vals, sizeof(double), nnz, fp); fread(b_arr, sizeof(double), N, fp);
  fclose(fp);
  Mat A;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscInt rstart, rend;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscInt *d_nnz = (PetscInt *)calloc(rend - rstart, sizeof(PetscInt));
  PetscInt *o_nnz = (PetscInt *)calloc(rend - rstart, sizeof(PetscInt));
  for (PetscInt i = rstart; i < rend; ++i) {
    int rs = ai[i], re = ai[i + 1];
    for (int kk = rs; kk < re; ++kk) {
      if (aj[kk] >= rstart && aj[kk] < rend) d_nnz[i - rstart]++;
      else                                    o_nnz[i - rstart]++;
    }
  }
  PetscCall(MatSeqAIJSetPreallocation(A, 0, d_nnz));
  PetscCall(MatMPIAIJSetPreallocation(A, 0, d_nnz, 0, o_nnz));
  free(d_nnz); free(o_nnz);
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = rstart; i < rend; ++i) {
    int rs = ai[i], re = ai[i + 1];
    for (int kk = rs; kk < re; ++kk)
      PetscCall(MatSetValue(A, i, aj[kk], vals[kk], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY));
  Vec b;
  PetscCall(VecCreate(comm, &b));
  PetscCall(VecSetType(b, VECMPI));
  PetscCall(VecSetSizes(b, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(b));
  for (PetscInt i = rstart; i < rend; ++i)
    PetscCall(VecSetValue(b, i, (PetscScalar)b_arr[i], INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  free(ai); free(aj); free(vals); free(b_arr);
  *A_out = A; *b_out = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Phase 4 final-audit — PC_SYMMETRIC rejection"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Mat A; Vec b;
  PetscCall(load_linsys_parallel(PETSC_COMM_WORLD, LINSYS, &A, &b));
  if (rank == 0) printf("[pc-symmetric-rejected] loaded cdr_small on %d ranks\n", (int)size);

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, 1e-10, PETSC_DEFAULT, 2000));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED));
  PetscCall(KSPSetPCSide(ksp, PC_SYMMETRIC));   /* the gate of this test */

  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));

  /* Disable PETSc's default error-on-failure abort so we can inspect the
     return code. PUSH the supress so we restore on cleanup. */
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));

  /* Try setup; this is where KSPSetSupportedNorm rejection should fire. */
  PetscErrorCode setup_err = KSPSetFromOptions(ksp);
  if (setup_err == 0) {
    setup_err = KSPSetUp(ksp);
  }

  /* If setup didn't reject, try the solve; the defense-in-depth check at
     the top of KSPSolve_GMSTAB should fire. */
  PetscErrorCode solve_err = 0;
  if (setup_err == 0) {
    solve_err = KSPSolve(ksp, b, x);
  }

  /* Restore default error handler before cleanup so our own cleanup doesn't
     silently swallow real errors. */
  PetscCall(PetscPopErrorHandler());

  int pass = 0;
  if (rank == 0) {
    printf("[pc-symmetric-rejected] setup_err=%d  solve_err=%d\n",
           (int)setup_err, (int)solve_err);
    /* PASS conditions: SOMETHING errored. We don't care which path catches
       it (registration vs. defense-in-depth), only that PC_SYMMETRIC does
       NOT silently complete with a wrong-answer x. */
    if (setup_err != 0 || solve_err != 0) {
      printf("[pc-symmetric-rejected] PASS — PC_SYMMETRIC was rejected as expected\n");
      pass = 1;
    } else {
      printf("[pc-symmetric-rejected] FAIL — PC_SYMMETRIC silently completed; "
             "this means EITHER (a) the registration drop was reverted, OR "
             "(b) the defense-in-depth PetscCheck at top of KSPSolve_GMSTAB "
             "was removed. Either way, users now get silent wrong-answer "
             "for PC_SYMMETRIC + any preconditioner.\n");
      pass = 0;
    }
  }
  PetscCallMPI(MPI_Bcast(&pass, 1, MPI_INT, 0, PETSC_COMM_WORLD));

  /* Cleanup. KSPDestroy may emit warnings about the failed setup; ignore. */
  KSPDestroy(&ksp);
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
