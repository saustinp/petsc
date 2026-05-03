/*
   Phase 4a validator — PC_RIGHT + PCJACOBI + KSPSetInitialGuessNonzero.

   What this gates that ex_gmstab_pcright_jacobi misses
   ----------------------------------------------------
   When KSPSetInitialGuessNonzero is on, gms->x_initial_guess captures
   the user's x0 at solve start (before x_local is zeroed in the algorithm).
   The Phase 4a unwrap is:
       x_user = x_initial + B⁻¹·(x_global + x_local − x_initial)
   so x_initial appears twice (once additively, once in the subtracted
   reference point inside the PCApply). If either save or restore is
   broken, the user-visible residual diverges by an amount proportional
   to ‖x_initial‖.

   With x0 = 0.7·𝟙 the test exercises:
     - solve-start save: VecCopy(x_global, x_initial_guess) inside the
       conditional pc_side block.
     - solve-end restore: PCApply on (x_global + x_local − x_initial)
       and VecAXPY add-back of x_initial in FinalizeSolution_Private.
     - per-snapshot trueres recompute: same unwrap inside Snapshot_Private,
       so that KSP-reported rnorm matches user-visible rnorm at each
       snapshot (and therefore the LAST one, which becomes ksp->rnorm).
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINSYS  "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"

#define TOLABS         1e-10
#define FINAL_RES_FUDGE 1.5

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
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Phase 4a — PC_RIGHT + Jacobi + nonzero initial guess"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Mat A; Vec b;
  PetscCall(load_linsys_parallel(PETSC_COMM_WORLD, LINSYS, &A, &b));
  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  if (rank == 0) printf("[pcright-nzg] loaded cdr_small: N=%" PetscInt_FMT " on %d ranks\n", N, (int)size);

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.7));   /* nonzero initial guess */

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, TOLABS, PETSC_DEFAULT, 2000));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));

  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  KSPConvergedReason reason;
  PetscReal          rnorm_internal;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm_internal));

  Vec Ax, r;
  PetscCall(VecDuplicate(b, &Ax));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(MatMult(A, x, Ax));
  PetscCall(VecWAXPY(r, -1.0, Ax, b));
  PetscReal user_visible_res;
  PetscCall(VecNorm(r, NORM_2, &user_visible_res));

  if (rank == 0) {
    printf("[pcright-nzg] x0=0.7·1, reason=%d rnorm_int=%.6e ext_res=%.6e (gate: <= %.6e)\n",
           (int)reason, (double)rnorm_internal, (double)user_visible_res,
           FINAL_RES_FUDGE * TOLABS);

    int g_reason = (reason == KSP_CONVERGED_ATOL);
    int g_user   = ((double)user_visible_res <= FINAL_RES_FUDGE * TOLABS);
    int g_self   = (fabs((double)user_visible_res - (double)rnorm_internal)
                      <= 1e-9 + 0.1 * fabs((double)rnorm_internal));

    printf("[pcright-nzg]   gate reason ATOL    : -> %s\n", g_reason ? "PASS" : "FAIL");
    printf("[pcright-nzg]   gate ext residual   : %.3e <= %.3e -> %s\n",
           (double)user_visible_res, FINAL_RES_FUDGE * TOLABS, g_user ? "PASS" : "FAIL");
    printf("[pcright-nzg]   gate self-consistent: |int-ext|=%.3e -> %s\n",
           fabs((double)user_visible_res - (double)rnorm_internal),
           g_self ? "PASS" : "FAIL");

    int pass = (g_reason && g_user && g_self) ? 1 : 0;
    printf("[pcright-nzg] %s\n", pass ? "[PASS]" : "[FAIL]");

    PetscCallMPI(MPI_Bcast(&pass, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCall(VecDestroy(&Ax));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(PetscFinalize());
    return pass ? 0 : 1;
  }
  int pass_recv = 1;
  PetscCallMPI(MPI_Bcast(&pass_recv, 1, MPI_INT, 0, PETSC_COMM_WORLD));
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return pass_recv ? 0 : 1;
}
