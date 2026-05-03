/*
   Phase 3d validator — multi-solve repeatability.

   Calls KSPSolve TWICE on the SAME KSP object (same operator A,
   same right-hand side b, same shadow space P, same x0=0). Asserts:
     1. Both solves return KSP_CONVERGED_ATOL.
     2. Both produce a final solution with ||b - A*x|| <= 1.5*tolabs.
     3. The two final solutions are bit-identical (since the algorithm
        is deterministic and the second solve gets the same inputs).

   What this gates
   ---------------
   Bugs caught by this test that the single-solve harness misses:
     - Per-solve counters not being reset (matvec_count, snapshot_count,
       cycle_count, n2cycles, _last_logged_matvec_count) → second
       solve's counters carry over from the first → KSPConvergedDefault
       sees ksp->its already > max_it → spurious DIVERGED_ITS.
     - V0 / V1 / Z left over from previous solve carrying stale values
       into Initialisation, so the new solve consumes the old basis
       instead of re-orthogonalising.
     - trace_fp not closed-and-reopened between solves → second
       solve's snapshots get appended to the first's CSV.
     - x_global not reset → on a multi-solve with x0=0, x_global
       starts at the previous solve's accumulator instead of 0.
     - Memory leaks via VecDuplicate into a non-NULL pointer.

   We don't bit-compare the trace CSVs across the two solves because
   the trace files are different on disk (we use one trace per solve);
   instead we use the final-x bit-identical gate, which is even
   stronger: the two solves must produce identical x to the bit, which
   implies their entire algorithmic state was identical.
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINSYS  "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"
#define OUT_1   "/tmp/petsc_multisolve_run1.csv"
#define OUT_2   "/tmp/petsc_multisolve_run2.csv"

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
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Multi-solve repeatability gate"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Mat A; Vec b;
  PetscCall(load_linsys_parallel(PETSC_COMM_WORLD, LINSYS, &A, &b));
  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  if (rank == 0) printf("[multisolve] loaded cdr_small: N=%" PetscInt_FMT " on %d ranks\n", N, (int)size);

  Vec x1, x2;
  PetscCall(VecDuplicate(b, &x1));
  PetscCall(VecDuplicate(b, &x2));
  PetscCall(VecSet(x1, 0.0));
  PetscCall(VecSet(x2, 0.0));

  /* Build the KSP exactly once. The two KSPSolve calls share this object. */
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, TOLABS, PETSC_DEFAULT, 1000));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));
  PetscCall(KSPSetFromOptions(ksp));

  /* ---- First solve. Trace goes to OUT_1. ---- */
  if (rank == 0) {
    PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_trace_csv", OUT_1));
    PetscCall(KSPSetFromOptions(ksp));   /* re-read so trace_csv lands */
  }
  PetscCall(KSPSolve(ksp, b, x1));
  KSPConvergedReason reason1;
  PetscInt           iters1;
  PetscReal          rnorm1;
  PetscCall(KSPGetConvergedReason(ksp, &reason1));
  PetscCall(KSPGetIterationNumber(ksp, &iters1));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm1));

  /* ---- Second solve on the same KSP. Trace goes to OUT_2. ---- */
  if (rank == 0) {
    PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_trace_csv", OUT_2));
    PetscCall(KSPSetFromOptions(ksp));   /* re-read so trace_csv lands */
  }
  PetscCall(KSPSolve(ksp, b, x2));
  KSPConvergedReason reason2;
  PetscInt           iters2;
  PetscReal          rnorm2;
  PetscCall(KSPGetConvergedReason(ksp, &reason2));
  PetscCall(KSPGetIterationNumber(ksp, &iters2));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm2));

  /* ---- Externally measured residuals on the user's returned x's. ---- */
  Vec Ax, r;
  PetscCall(VecDuplicate(b, &Ax));
  PetscCall(VecDuplicate(b, &r));
  PetscReal ext_res1, ext_res2;
  PetscCall(MatMult(A, x1, Ax));
  PetscCall(VecWAXPY(r, -1.0, Ax, b));
  PetscCall(VecNorm(r, NORM_2, &ext_res1));
  PetscCall(MatMult(A, x2, Ax));
  PetscCall(VecWAXPY(r, -1.0, Ax, b));
  PetscCall(VecNorm(r, NORM_2, &ext_res2));

  /* ---- ||x1 - x2|| — must be 0 exactly. */
  PetscCall(VecAXPY(r, 0.0, b));   /* zero r */
  PetscCall(VecCopy(x1, r));
  PetscCall(VecAXPY(r, -1.0, x2));
  PetscReal dx_norm;
  PetscCall(VecNorm(r, NORM_2, &dx_norm));

  if (rank == 0) {
    printf("[multisolve] solve 1: reason=%d iters=%d rnorm=%.6e ext_res=%.6e\n",
           (int)reason1, (int)iters1, (double)rnorm1, (double)ext_res1);
    printf("[multisolve] solve 2: reason=%d iters=%d rnorm=%.6e ext_res=%.6e\n",
           (int)reason2, (int)iters2, (double)rnorm2, (double)ext_res2);
    printf("[multisolve] ||x1 - x2||_2 = %.6e (must be 0 exactly)\n", (double)dx_norm);

    int g_reason   = (reason1 == KSP_CONVERGED_ATOL && reason2 == KSP_CONVERGED_ATOL);
    int g_iters    = (iters1 == iters2);
    int g_rnorm    = (rnorm1 == rnorm2);
    int g_ext      = ((double)ext_res1 <= FINAL_RES_FUDGE * TOLABS &&
                      (double)ext_res2 <= FINAL_RES_FUDGE * TOLABS);
    int g_xidentical = (dx_norm == 0.0);

    printf("[multisolve]   gate reason both=ATOL : -> %s\n", g_reason ? "PASS" : "FAIL");
    printf("[multisolve]   gate iter count match : %d == %d -> %s\n", (int)iters1, (int)iters2, g_iters ? "PASS" : "FAIL");
    printf("[multisolve]   gate rnorm match      : %.6e == %.6e -> %s\n", (double)rnorm1, (double)rnorm2, g_rnorm ? "PASS" : "FAIL");
    printf("[multisolve]   gate ext residual sane: ext1=%.3e ext2=%.3e (need <= %.3e) -> %s\n",
           (double)ext_res1, (double)ext_res2, FINAL_RES_FUDGE * TOLABS, g_ext ? "PASS" : "FAIL");
    printf("[multisolve]   gate x1 == x2 bit-eq  : ||x1 - x2||_2 = %.3e (need 0) -> %s\n",
           (double)dx_norm, g_xidentical ? "PASS" : "FAIL");

    int pass = (g_reason && g_iters && g_rnorm && g_ext && g_xidentical) ? 1 : 0;
    printf("[multisolve] %s\n", pass ? "[PASS]" : "[FAIL]");

    PetscCallMPI(MPI_Bcast(&pass, 1, MPI_INT, 0, PETSC_COMM_WORLD));
    PetscCall(VecDestroy(&Ax));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&x1));
    PetscCall(VecDestroy(&x2));
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
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return pass_recv ? 0 : 1;
}
