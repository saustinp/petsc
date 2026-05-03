/*
   Phase 3d validator — full natural-flow with KSPSetInitialGuessNonzero.

   What this gates
   ---------------
   The natural-flow driver maintains x_global (the running cumulative
   restart accumulator) and x_local (the current cycle-local update).
   On every return path, the user-visible solution is x_global +
   x_local. For zero initial guess (the bit-equivalence harness)
   x_global stays 0 throughout if no restart fires, so the bug is
   masked. This validator exercises the OTHER paths.

   Specifically:
     1. Pick a nonzero initial guess x0 (constant vector, easy to
        verify by inspection).
     2. Run the full natural-flow KSPSolve to convergence.
     3. Assert ||b - A*x_final||_2 <= 1.5*tolabs — i.e., the user
        gets a *correct* solution, not the cycle-local x_local left
        over from the last restart.

   Failure modes this catches
     - x_local += x_global accumulation skipped on early-out (the
       force_l*_only branches and the post-init early-out paths
       previously had this bug; without the fix, x_local on return
       would be a partial cycle update, not the full solution).
     - Wrong sign / wrong scaling in the accumulation.

   Run modes: sequential and parallel (n=2,4,8). Parallel uses the
   same correctness check (norm-based, so MPI Allreduce reorder is
   absorbed by the 1.5*tolabs slack).
*/

#include <petscksp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define LINSYS "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN   "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"

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
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Natural-flow with nonzero initial guess"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Mat A; Vec b;
  PetscCall(load_linsys_parallel(PETSC_COMM_WORLD, LINSYS, &A, &b));
  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  if (rank == 0) printf("[nzg-natural] loaded cdr_small: N=%" PetscInt_FMT " on %d ranks\n", N, (int)size);

  /* x0 = constant 0.7 — clearly nonzero, well-defined under VecSet,
     trivially the same on every rank. */
  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.7));

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, TOLABS, PETSC_DEFAULT, 1000));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  KSPConvergedReason reason;
  PetscInt           iters;
  PetscReal          rnorm_internal;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetIterationNumber(ksp, &iters));
  PetscCall(KSPGetResidualNorm(ksp, &rnorm_internal));

  /* THE GATE: ||b - A * x_final||_2 must be <= 1.5*TOLABS.
     This is the only thing that matters — the user's x must be a
     solution to A*x = b, regardless of how the algorithm got there. */
  Vec Ax, r;
  PetscCall(VecDuplicate(b, &Ax));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(MatMult(A, x, Ax));
  PetscCall(VecWAXPY(r, -1.0, Ax, b));
  PetscReal user_visible_res;
  PetscCall(VecNorm(r, NORM_2, &user_visible_res));

  if (rank == 0) {
    printf("[nzg-natural] x0 = 0.7 (constant), ||b||=?, KSPSolve reason=%d iters=%d\n",
           (int)reason, (int)iters);
    printf("[nzg-natural] internal rnorm reported by KSP   = %.6e\n", (double)rnorm_internal);
    printf("[nzg-natural] user-visible ||b - A*x_returned|| = %.6e (gate: <= %.6e)\n",
           (double)user_visible_res, FINAL_RES_FUDGE * TOLABS);
  }

  int pass = 1;
  if (rank == 0) {
    int gate_reason = (reason == KSP_CONVERGED_ATOL);
    int gate_user   = ((double)user_visible_res <= FINAL_RES_FUDGE * TOLABS);
    /* Self-consistency check: KSP's reported rnorm and the externally
       measured rnorm must agree to FP precision. If they diverge, it
       means KSP is reporting a residual computed against a *different*
       x than what's stored in ksp->vec_sol — i.e., the x_global
       accumulation bug.  If KSP claims residual 5e-11 but the true
       residual against the returned x is 0.7, that's the bug. */
    int gate_self_consistent = (fabs((double)user_visible_res - (double)rnorm_internal)
                                  <= 1e-9 + 0.1 * fabs((double)rnorm_internal));
    printf("[nzg-natural]   gate reason            : %d=KSP_CONVERGED_ATOL -> %s\n",
           (int)reason, gate_reason ? "PASS" : "FAIL");
    printf("[nzg-natural]   gate user-visible res  : %.3e <= %.3e -> %s\n",
           (double)user_visible_res, FINAL_RES_FUDGE * TOLABS, gate_user ? "PASS" : "FAIL");
    printf("[nzg-natural]   gate self-consistent   : |internal-external| = %.3e -> %s\n",
           fabs((double)user_visible_res - (double)rnorm_internal),
           gate_self_consistent ? "PASS" : "FAIL");
    pass = (gate_reason && gate_user && gate_self_consistent) ? 1 : 0;
    printf("[nzg-natural] %s\n", pass ? "[PASS]" : "[FAIL]");
  }
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
