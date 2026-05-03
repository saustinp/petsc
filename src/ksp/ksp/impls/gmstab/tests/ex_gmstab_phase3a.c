/*
   Phase 3a validator — runs KSPGMSTAB on cdr_small with deterministic
   shadow space P from P.bin and compares the first 2 rows of the
   trace CSV against the C++ port's output.

   Expected: the iter=0 (init constructor) and iter=1 (post-Initialisation)
   rows must match the C++ port's residual snapshots to <= 1e-10
   absolute drift on iterres and trueres, and matvec exactly.

   Build:
     /home/sam/.local/mpich/bin/mpicc \
       -I/home/sam/hpc_stack/petsc/include \
       -I/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/include \
       /home/sam/hpc_stack/petsc/src/ksp/ksp/impls/gmstab/tests/ex_gmstab_phase3a.c \
       -L/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/lib \
       -Wl,-rpath,/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/lib \
       -lpetsc -lm -o /tmp/ex_gmstab_phase3a

   Run:
     /tmp/ex_gmstab_phase3a
*/

#include <petscksp.h>
#include <stdio.h>
#include <string.h>

#define LINSYS "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN   "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"
#define REF    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/cpp_residuals.csv"

/* Read EXASIMLS .bin into a PETSc Mat + Vec on PETSC_COMM_SELF.
   Header layout per gmstab_handoff/csr_format_instructions.md:
     0..7       8-byte magic "EXASIMLS"
     8..15      int32 version, int32 index_dtype_size
     16..23     int32 value_dtype_size, int32 reserved
     24..31     uint64 N (rows)
     32..39     uint64 nnz
     40..63     reserved
   Then: rowptr (N+1), colidx (nnz), values (nnz, double), b (N, double).
   For cdr_small: index_dtype_size=4 (int32), value_dtype_size=8 (float64). */
static PetscErrorCode load_linsys(const char *path, Mat *A_out, Vec *b_out)
{
  FILE *fp = fopen(path, "rb");
  PetscFunctionBegin;
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s", path);

  char magic[8];
  fread(magic, 1, 8, fp);
  PetscCheck(memcmp(magic, "EXASIMLS", 8) == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "magic");

  int version, idx_size, val_size, reserved;
  fread(&version, sizeof(int), 1, fp);
  fread(&idx_size, sizeof(int), 1, fp);
  fread(&val_size, sizeof(int), 1, fp);
  fread(&reserved, sizeof(int), 1, fp);

  unsigned long long N_u, nnz_u;
  fread(&N_u, 8, 1, fp);
  fread(&nnz_u, 8, 1, fp);
  fseek(fp, 64, SEEK_SET);

  const PetscInt N   = (PetscInt)N_u;
  const PetscInt nnz = (PetscInt)nnz_u;

  /* Read rowptr, colidx (int32 in cdr_small), values (double), b (double). */
  int *ai_int32 = (int *)malloc(sizeof(int) * (N + 1));
  int *aj_int32 = (int *)malloc(sizeof(int) * nnz);
  double *vals = (double *)malloc(sizeof(double) * nnz);
  double *b_arr = (double *)malloc(sizeof(double) * N);

  PetscCheck(idx_size == 4, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "phase 3a: idx_size != 4 (got %d)", idx_size);

  fread(ai_int32, sizeof(int), N + 1, fp);
  fread(aj_int32, sizeof(int), nnz, fp);
  fread(vals, sizeof(double), nnz, fp);
  fread(b_arr, sizeof(double), N, fp);
  fclose(fp);

  /* Build a sequential AIJ matrix. */
  Mat A;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSetSizes(A, N, N, N, N));
  /* Pre-count nz/row for proper preallocation. */
  PetscInt *nnz_per_row = (PetscInt *)malloc(sizeof(PetscInt) * N);
  for (PetscInt i = 0; i < N; ++i) nnz_per_row[i] = ai_int32[i + 1] - ai_int32[i];
  PetscCall(MatSeqAIJSetPreallocation(A, 0, nnz_per_row));
  free(nnz_per_row);
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = 0; i < N; ++i) {
    int rowstart = ai_int32[i];
    int rowend = ai_int32[i + 1];
    for (int kk = rowstart; kk < rowend; ++kk) {
      PetscCall(MatSetValue(A, i, aj_int32[kk], vals[kk], INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY));

  Vec b;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &b));
  PetscScalar *bptr;
  PetscCall(VecGetArray(b, &bptr));
  for (PetscInt i = 0; i < N; ++i) bptr[i] = (PetscScalar)b_arr[i];
  PetscCall(VecRestoreArray(b, &bptr));

  free(ai_int32);
  free(aj_int32);
  free(vals);
  free(b_arr);

  *A_out = A;
  *b_out = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Phase 3a Initialisation validator"));

  Mat A; Vec b;
  PetscCall(load_linsys(LINSYS, &A, &b));

  PetscInt N;
  PetscCall(MatGetSize(A, &N, NULL));
  printf("[phase3a] loaded cdr_small: N=%" PetscInt_FMT "\n", N);

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));

  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, 1e-10, PETSC_DEFAULT, 500));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  /* Configure shadow space + trace via options DB. */
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s", "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_trace_csv", "/tmp/petsc_phase3a_residuals.csv"));
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve — expect KSP_DIVERGED_BREAKDOWN at end of Initialisation
     (cycle bodies are not yet implemented). */
  PetscCall(KSPSolve(ksp, b, x));

  KSPConvergedReason reason;
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  printf("[phase3a] KSPSolve returned reason=%d (expected -10/DIVERGED_BREAKDOWN or 2/CONVERGED_ATOL)\n", (int)reason);

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /* Compare /tmp/petsc_phase3a_residuals.csv against REF (first 2 rows). */
  FILE *ours = fopen("/tmp/petsc_phase3a_residuals.csv", "r");
  FILE *ref  = fopen(REF, "r");
  if (!ours || !ref) {
    printf("[phase3a] cannot open files for comparison\n");
    return 1;
  }
  char buf_o[1024], buf_r[1024];
  fgets(buf_o, sizeof(buf_o), ours);  /* header */
  fgets(buf_r, sizeof(buf_r), ref);

  int rows_compared = 0;
  int matvec_match  = 1;
  double max_iter_drift = 0.0, max_true_drift = 0.0;
  int constructor_strict_pass = 1;
  for (int row = 0; row < 2 && fgets(buf_o, sizeof(buf_o), ours) && fgets(buf_r, sizeof(buf_r), ref); ++row) {
    int io, mo;
    double iro, tro, ru_o, rmv_o;
    int ir_, mr;
    double irr, trr, ru_r, rmv_r;
    sscanf(buf_o, "%d,%d,%lf,%lf,%lf,%lf", &io, &mo, &iro, &tro, &ru_o, &rmv_o);
    sscanf(buf_r, "%d,%d,%lf,%lf,%lf,%lf", &ir_, &mr, &irr, &trr, &ru_r, &rmv_r);
    if (mo != mr) matvec_match = 0;
    double dr_iter = fabs(iro - irr);
    double dr_true = fabs(tro - trr);
    if (dr_iter > max_iter_drift) max_iter_drift = dr_iter;
    if (dr_true > max_true_drift) max_true_drift = dr_true;
    printf("[phase3a] row %d: ours=(matvec=%d, iter=%.6e, true=%.6e)  ref=(matvec=%d, iter=%.6e, true=%.6e)  drift=%.2e/%.2e\n",
           row, mo, iro, tro, mr, irr, trr, dr_iter, dr_true);

    /* Strict constructor-row checks (tripwire — catches the regressions we
       worked through in Phase 3b). The constructor row must satisfy:
         iter == 0, matvec == 0
         iterres == trueres (within machine epsilon — both should be ||b||)
         iter and matvec runtimes both 0 (no work has happened yet)
       These are enforced strictly because any drift here means a
       non-trivial state change snuck into the constructor path. */
    if (row == 0) {
      if (io != 0 || mo != 0) {
        printf("[phase3a]   STRICT FAIL: constructor row must have iter=0,mv=0 (got iter=%d,mv=%d)\n", io, mo);
        constructor_strict_pass = 0;
      }
      if (fabs(iro - tro) > 1e-15) {
        printf("[phase3a]   STRICT FAIL: constructor iterres != trueres (|diff|=%.3e)\n", fabs(iro - tro));
        constructor_strict_pass = 0;
      }
      if (ru_o != 0.0 || rmv_o != 0.0) {
        printf("[phase3a]   STRICT FAIL: constructor runtimes must be 0 (got %.3e/%.3e)\n", ru_o, rmv_o);
        constructor_strict_pass = 0;
      }
    }
    rows_compared++;
  }
  fclose(ours);
  fclose(ref);

  printf("[phase3a] rows_compared=%d matvec_match=%d max_iter_drift=%.3e max_true_drift=%.3e constructor_strict=%d\n",
         rows_compared, matvec_match, max_iter_drift, max_true_drift, constructor_strict_pass);
  int pass = (matvec_match
              && max_iter_drift < 1e-10
              && max_true_drift < 1e-10
              && constructor_strict_pass
              && rows_compared == 2) ? 1 : 0;
  printf("[phase3a] %s\n", pass ? "[PASS]" : "[FAIL]");

  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
