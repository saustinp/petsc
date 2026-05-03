/*
   ex_gmstab_init_nonzero_guess — exercises the non-zero-initial-guess path
   through KSPSolve_GMSTAB. The point is to assert that the constructor
   snapshot still reports trueres = ||b|| even when x0 != 0, matching the
   C++ port's PerfMeasure constructor semantics.

   Without the special-case in KSPGMSTABSnapshot_Private, the constructor
   trueres would equal ||b - A*x0||, which is wrong.

   This is a tripwire: it does not need to converge — it only checks that
   the constructor row in the trace CSV satisfies:
       iter == 0
       matvec == 0
       iterres == trueres == ||b||  (within 1e-12)
   regardless of the user's initial guess.

   Build: same flags as ex_gmstab_phase3a.
   Run:   /tmp/ex_gmstab_init_nonzero_guess
*/

#include <petscksp.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define LINSYS "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN   "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"
#define OUT    "/tmp/petsc_init_nonzero_residuals.csv"

static PetscErrorCode load_linsys(const char *path, Mat *A_out, Vec *b_out)
{
  FILE *fp = fopen(path, "rb");
  PetscFunctionBegin;
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", path);
  char magic[8];
  fread(magic, 1, 8, fp);
  PetscCheck(memcmp(magic, "EXASIMLS", 8) == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "magic");
  int version, idx_size, val_size, reserved;
  fread(&version, sizeof(int), 1, fp); fread(&idx_size, sizeof(int), 1, fp);
  fread(&val_size, sizeof(int), 1, fp); fread(&reserved, sizeof(int), 1, fp);
  unsigned long long N_u, nnz_u;
  fread(&N_u, 8, 1, fp); fread(&nnz_u, 8, 1, fp);
  fseek(fp, 64, SEEK_SET);
  const PetscInt N   = (PetscInt)N_u;
  const PetscInt nnz = (PetscInt)nnz_u;
  PetscCheck(idx_size == 4, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "idx_size != 4");
  int *ai = (int *)malloc(sizeof(int) * (N + 1));
  int *aj = (int *)malloc(sizeof(int) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *b_arr = (double *)malloc(sizeof(double) * N);
  fread(ai, sizeof(int), N + 1, fp); fread(aj, sizeof(int), nnz, fp);
  fread(vals, sizeof(double), nnz, fp); fread(b_arr, sizeof(double), N, fp);
  fclose(fp);
  Mat A;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSetSizes(A, N, N, N, N));
  PetscInt *npr = (PetscInt *)malloc(sizeof(PetscInt) * N);
  for (PetscInt i = 0; i < N; ++i) npr[i] = ai[i + 1] - ai[i];
  PetscCall(MatSeqAIJSetPreallocation(A, 0, npr));
  free(npr);
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = 0; i < N; ++i) {
    int rs = ai[i], re = ai[i + 1];
    for (int kk = rs; kk < re; ++kk) PetscCall(MatSetValue(A, i, aj[kk], vals[kk], INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY));
  Vec b;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &b));
  PetscScalar *bp;
  PetscCall(VecGetArray(b, &bp));
  for (PetscInt i = 0; i < N; ++i) bp[i] = (PetscScalar)b_arr[i];
  PetscCall(VecRestoreArray(b, &bp));
  free(ai); free(aj); free(vals); free(b_arr);
  *A_out = A; *b_out = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Init non-zero initial-guess tripwire"));

  Mat A; Vec b;
  PetscCall(load_linsys(LINSYS, &A, &b));
  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  printf("[init-nzg] loaded cdr_small: N=%" PetscInt_FMT "\n", N);

  PetscReal nb;
  PetscCall(VecNorm(b, NORM_2, &nb));

  /* Set x to a nonzero arbitrary guess; this must NOT change the
     constructor row's trueres column (must remain ||b||). */
  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 1.7));   /* arbitrary, decidedly non-zero */

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, 1e-10, PETSC_DEFAULT, 500));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",         "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file",    PBIN));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_trace_csv", OUT));
  PetscCall(KSPSetFromOptions(ksp));

  /* We don't care about convergence; we only care about the constructor row. */
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /* Read the trace and validate the constructor row. */
  FILE *ours = fopen(OUT, "r");
  if (!ours) { printf("[init-nzg] cannot open %s\n", OUT); return 1; }
  char buf[1024];
  fgets(buf, sizeof(buf), ours);  /* header */
  if (!fgets(buf, sizeof(buf), ours)) {
    printf("[init-nzg] no rows in trace\n");
    fclose(ours);
    return 1;
  }
  int it_, mv_;
  double iro, tro, ru, rmv;
  sscanf(buf, "%d,%d,%lf,%lf,%lf,%lf", &it_, &mv_, &iro, &tro, &ru, &rmv);
  fclose(ours);

  printf("[init-nzg] x0 = 1.7*ones(N), ||b|| = %.16e\n", nb);
  printf("[init-nzg] constructor row: iter=%d, mv=%d, iterres=%.16e, trueres=%.16e, runtime=%.3e/%.3e\n",
         it_, mv_, iro, tro, ru, rmv);

  int pass = 1;
  if (it_ != 0 || mv_ != 0) {
    printf("[init-nzg] FAIL: constructor row must be (iter=0, mv=0)\n");
    pass = 0;
  }
  if (fabs(iro - nb) > 1e-12) {
    printf("[init-nzg] FAIL: constructor iterres %.6e != ||b|| %.6e (|diff|=%.3e)\n", iro, nb, fabs(iro - nb));
    pass = 0;
  }
  if (fabs(tro - nb) > 1e-12) {
    printf("[init-nzg] FAIL: constructor trueres %.6e != ||b|| %.6e (|diff|=%.3e) — "
           "the C++ port reports ||b|| even with non-zero x0; "
           "did the Snapshot_Private constructor special-case regress?\n",
           tro, nb, fabs(tro - nb));
    pass = 0;
  }
  if (ru != 0.0 || rmv != 0.0) {
    printf("[init-nzg] FAIL: constructor runtimes must be 0\n");
    pass = 0;
  }
  printf("[init-nzg] %s\n", pass ? "[PASS]" : "[FAIL]");

  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
