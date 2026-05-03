/*
   Phase 4 audit validator — multi-solve interactions with PC_RIGHT.

   Exercises three multi-solve scenarios that the per-PC tripwires
   (ex_gmstab_pcright_jacobi, _pcleft_jacobi, etc.) and the existing
   ex_gmstab_multisolve (PCNONE) don't cover:

     1. Two PC_RIGHT solves on the same KSP with the SAME nonzero
        initial guess. Asserts both solves are bit-identical (same
        rnorm, same x). Verifies that x_initial_guess is correctly
        re-allocated per-solve and that the unwrap algebra is
        deterministic across re-solves.

     2. Two PC_RIGHT solves with DIFFERENT nonzero initial guesses.
        Asserts both produce valid solutions (‖b−A·x‖ ≤ 1.5·tolabs).
        Verifies x_initial_guess capture is per-solve (Pattern A: not
        carried over from prior solve).

     3. PC_RIGHT solve, then KSPSetPCSide(PC_LEFT), then second solve.
        Asserts second solve produces a valid solution. Verifies that
        the destroy-then-conditional-allocate logic for x_initial_guess
        correctly drops the field on a pc_side downgrade.

   These three scenarios target the most likely Phase 4a regressions
   in production usage (Newton iterations, library users that change
   PC mid-flight, etc.).
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINSYS  "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"

/* See ex_gmstab_pcleft_jacobi.c — TOLABS drives algorithm short-circuit on
   ||r_pre||; GATE_TOL is the looser user-visible bound that absorbs the
   conditioning-ratio stall under PC_LEFT + KSP_NORM_UNPRECONDITIONED. */
#define TOLABS         1e-10
#define GATE_TOL       1e-8
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

/* Solve once and return: ksp->reason, ksp->rnorm, externally-measured
   ‖b - A·x‖. Caller owns x; we don't reset it. */
static PetscErrorCode solve_one(KSP ksp, Mat A, Vec b, Vec x,
                                KSPConvergedReason *reason_out, PetscReal *rnorm_int_out,
                                PetscReal *ext_res_out)
{
  PetscFunctionBegin;
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetConvergedReason(ksp, reason_out));
  PetscCall(KSPGetResidualNorm(ksp, rnorm_int_out));

  Vec Ax, r;
  PetscCall(VecDuplicate(b, &Ax));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(MatMult(A, x, Ax));
  PetscCall(VecWAXPY(r, -1.0, Ax, b));
  PetscCall(VecNorm(r, NORM_2, ext_res_out));
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Phase 4 audit — PC multi-solve scenarios"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  Mat A; Vec b;
  PetscCall(load_linsys_parallel(PETSC_COMM_WORLD, LINSYS, &A, &b));
  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  if (rank == 0) printf("[pc-multisolve] cdr_small N=%" PetscInt_FMT " on %d ranks\n\n",
                        N, (int)size);

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));

  /* --- Scenario 1: two PC_RIGHT solves with identical x0 = 0.7·𝟙. --- */
  /* The expected behavior is bit-identical results: x_initial_guess
     captured each solve, deterministic algorithm, identical inputs →
     identical x. */
  int s1_pass = 1;
  if (rank == 0) printf("  --- Scenario 1: PC_RIGHT × 2 with identical x0 = 0.7·1 ---\n");
  {
    Vec x1, x2;
    PetscCall(VecDuplicate(b, &x1));
    PetscCall(VecDuplicate(b, &x2));

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
    PetscCall(KSPSetFromOptions(ksp));

    KSPConvergedReason r1, r2;
    PetscReal rn1, rn2, ext1, ext2;

    PetscCall(VecSet(x1, 0.7));
    PetscCall(solve_one(ksp, A, b, x1, &r1, &rn1, &ext1));

    PetscCall(VecSet(x2, 0.7));
    PetscCall(solve_one(ksp, A, b, x2, &r2, &rn2, &ext2));

    Vec dx;
    PetscCall(VecDuplicate(b, &dx));
    PetscCall(VecCopy(x1, dx));
    PetscCall(VecAXPY(dx, -1.0, x2));
    PetscReal dx_norm;
    PetscCall(VecNorm(dx, NORM_2, &dx_norm));
    PetscCall(VecDestroy(&dx));

    int g_r1 = (r1 == KSP_CONVERGED_ATOL) && (ext1 <= FINAL_RES_FUDGE * TOLABS);
    int g_r2 = (r2 == KSP_CONVERGED_ATOL) && (ext2 <= FINAL_RES_FUDGE * TOLABS);
    int g_match = (rn1 == rn2) && (dx_norm == 0.0);

    if (rank == 0) {
      printf("    solve 1: reason=%d ext_res=%.3e\n", (int)r1, (double)ext1);
      printf("    solve 2: reason=%d ext_res=%.3e\n", (int)r2, (double)ext2);
      printf("    rn1==rn2 ? %s   ‖x1-x2‖=%.3e\n", (rn1 == rn2) ? "YES" : "NO", (double)dx_norm);
      printf("    Scenario 1: %s%s%s\n",
             g_r1 ? "convergence-OK " : "CONVERGENCE-FAIL ",
             g_match ? "match-OK " : "MATCH-FAIL ",
             (g_r1 && g_r2 && g_match) ? "[PASS]" : "[FAIL]");
    }
    s1_pass = (g_r1 && g_r2 && g_match) ? 1 : 0;

    PetscCall(VecDestroy(&x1));
    PetscCall(VecDestroy(&x2));
    PetscCall(KSPDestroy(&ksp));
  }

  /* --- Scenario 2: two PC_RIGHT solves with DIFFERENT x0. --- */
  int s2_pass = 1;
  if (rank == 0) printf("\n  --- Scenario 2: PC_RIGHT × 2 with x0_1 = 0.5·1, x0_2 = 1.3·1 ---\n");
  {
    Vec x1, x2;
    PetscCall(VecDuplicate(b, &x1));
    PetscCall(VecDuplicate(b, &x2));

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
    PetscCall(KSPSetFromOptions(ksp));

    KSPConvergedReason r1, r2;
    PetscReal rn1, rn2, ext1, ext2;

    PetscCall(VecSet(x1, 0.5));
    PetscCall(solve_one(ksp, A, b, x1, &r1, &rn1, &ext1));

    PetscCall(VecSet(x2, 1.3));
    PetscCall(solve_one(ksp, A, b, x2, &r2, &rn2, &ext2));

    int g1 = (r1 == KSP_CONVERGED_ATOL) && (ext1 <= FINAL_RES_FUDGE * TOLABS);
    int g2 = (r2 == KSP_CONVERGED_ATOL) && (ext2 <= FINAL_RES_FUDGE * TOLABS);

    if (rank == 0) {
      printf("    solve 1 (x0=0.5): reason=%d ext_res=%.3e\n", (int)r1, (double)ext1);
      printf("    solve 2 (x0=1.3): reason=%d ext_res=%.3e\n", (int)r2, (double)ext2);
      printf("    Scenario 2: %s\n", (g1 && g2) ? "[PASS]" : "[FAIL]");
    }
    s2_pass = (g1 && g2) ? 1 : 0;

    PetscCall(VecDestroy(&x1));
    PetscCall(VecDestroy(&x2));
    PetscCall(KSPDestroy(&ksp));
  }

  /* --- Scenario 3: PC_RIGHT, then PC_LEFT on same KSP. --- */
  int s3_pass = 1;
  if (rank == 0) printf("\n  --- Scenario 3: PC_RIGHT solve, switch to PC_LEFT, second solve ---\n");
  {
    Vec x1, x2;
    PetscCall(VecDuplicate(b, &x1));
    PetscCall(VecDuplicate(b, &x2));

    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPGMSTAB));
    PetscCall(KSPSetTolerances(ksp, 0.0, TOLABS, PETSC_DEFAULT, 2000));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));

    KSPConvergedReason r1, r2;
    PetscReal rn1, rn2, ext1, ext2;

    PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(VecSet(x1, 0.7));
    PetscCall(solve_one(ksp, A, b, x1, &r1, &rn1, &ext1));

    /* Switch pc_side. The destroy-at-top-of-KSPSolve drops gms->x_initial_guess
       (allocated for PC_RIGHT) and the conditional save skips it for PC_LEFT. */
    PetscCall(KSPSetPCSide(ksp, PC_LEFT));
    PetscCall(VecSet(x2, 0.0));        /* zero guess; PC_LEFT path */
    PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
    PetscCall(solve_one(ksp, A, b, x2, &r2, &rn2, &ext2));

    int g1 = (r1 == KSP_CONVERGED_ATOL) && (ext1 <= FINAL_RES_FUDGE * TOLABS);
    /* PC_LEFT under KSP_NORM_UNPRECONDITIONED: see ex_gmstab_pcleft_jacobi.c
       — accept ATOL or DIVERGED_ITS, gate the user-visible residual against
       the looser GATE_TOL to absorb the conditioning-ratio stall. */
    int g2 = ((r2 == KSP_CONVERGED_ATOL || r2 == KSP_DIVERGED_ITS)
              && (ext2 <= FINAL_RES_FUDGE * GATE_TOL));

    if (rank == 0) {
      printf("    solve 1 (PC_RIGHT, x0=0.7·1): reason=%d ext_res=%.3e\n",
             (int)r1, (double)ext1);
      printf("    solve 2 (PC_LEFT,  x0=0):     reason=%d ext_res=%.3e\n",
             (int)r2, (double)ext2);
      printf("    Scenario 3: %s\n", (g1 && g2) ? "[PASS]" : "[FAIL]");
    }
    s3_pass = (g1 && g2) ? 1 : 0;

    PetscCall(VecDestroy(&x1));
    PetscCall(VecDestroy(&x2));
    PetscCall(KSPDestroy(&ksp));
  }

  int pass = (s1_pass && s2_pass && s3_pass) ? 1 : 0;
  if (rank == 0) {
    printf("\n[pc-multisolve] %s\n", pass ? "[PASS]" : "[FAIL]");
  }
  PetscCallMPI(MPI_Bcast(&pass, 1, MPI_INT, 0, PETSC_COMM_WORLD));

  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
