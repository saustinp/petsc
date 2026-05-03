/*
   Phase 3d validator — same-rank determinism.

   Calls KSPSolve_GMSTAB twice on the same KSP/operator (with the
   trace_csv option pointing at two different paths so we can compare
   them) and asserts the two traces are BIT-IDENTICAL byte-for-byte.

   What this gates
   ---------------
   The full Krylov method must be a deterministic function of (A, b,
   shadow_space P, tolabs, max_it, rank_count). Any non-determinism
   (uninitialised memory, RNG with timing-based seed, raced MPI
   allreduces, etc.) would cause the two traces to diverge.

   PETSc's MPI_Allreduce is bitwise-reproducible on the same rank
   count with the same input, and our LAPACK calls are deterministic,
   so the only way to fail this gate is a real implementation bug.

   The test runs at every supported rank count via mpiexec.
*/

#include <petscksp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINSYS  "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/linsys.bin"
#define PBIN    "/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines/cdr_small/P.bin"
#define OUT_A   "/tmp/petsc_determinism_run_A.csv"
#define OUT_B   "/tmp/petsc_determinism_run_B.csv"

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

/* Run one full natural-flow KSPSolve into a fresh KSP, write the trace
   to `trace_path`, return the iteration count. */
static PetscErrorCode run_one(MPI_Comm comm, const char *trace_path, PetscMPIInt rank,
                              PetscInt *iters_out)
{
  PetscFunctionBegin;
  Mat A; Vec b;
  PetscCall(load_linsys_parallel(comm, LINSYS, &A, &b));

  Vec x;
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecSet(x, 0.0));

  KSP ksp;
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPGMSTAB));
  PetscCall(KSPSetTolerances(ksp, 0.0, 1e-10, PETSC_DEFAULT, 1000));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_s",      "4"));
  PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_p_file", PBIN));
  if (rank == 0) {
    PetscCall(PetscOptionsSetValue(NULL, "-ksp_gmstab_trace_csv", trace_path));
  }
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, iters_out));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Byte-by-byte file diff. Returns 1 if files match, 0 otherwise.
   Also reports total bytes compared and the first divergence byte. */
static int byte_diff(const char *a_path, const char *b_path, long *bytes_compared,
                     long *first_diff_byte)
{
  FILE *fa = fopen(a_path, "rb");
  FILE *fb = fopen(b_path, "rb");
  if (!fa || !fb) {
    if (fa) fclose(fa);
    if (fb) fclose(fb);
    *bytes_compared = -1;
    *first_diff_byte = -1;
    return 0;
  }
  long pos = 0;
  int identical = 1;
  *first_diff_byte = -1;
  while (1) {
    unsigned char ca, cb;
    size_t na = fread(&ca, 1, 1, fa);
    size_t nb = fread(&cb, 1, 1, fb);
    if (na == 0 && nb == 0) break;
    if (na != nb) {
      identical = 0;
      if (*first_diff_byte < 0) *first_diff_byte = pos;
      break;
    }
    if (ca != cb) {
      identical = 0;
      if (*first_diff_byte < 0) *first_diff_byte = pos;
      break;
    }
    pos++;
  }
  *bytes_compared = pos;
  fclose(fa);
  fclose(fb);
  return identical;
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, "Same-rank determinism gate"));

  PetscMPIInt rank, size;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* Wipe any stale traces so an old run can't masquerade as a current
     one. Only rank 0 owns the trace files. */
  if (rank == 0) {
    remove(OUT_A);
    remove(OUT_B);
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  PetscInt iters_A, iters_B;
  PetscCall(run_one(PETSC_COMM_WORLD, OUT_A, rank, &iters_A));
  PetscCall(run_one(PETSC_COMM_WORLD, OUT_B, rank, &iters_B));

  if (rank == 0) {
    printf("[determinism] size=%d  run-A iters=%d  run-B iters=%d\n",
           (int)size, (int)iters_A, (int)iters_B);
  }

  int pass = 1;
  if (rank == 0) {
    long bytes = 0, first_diff = -1;
    int identical = byte_diff(OUT_A, OUT_B, &bytes, &first_diff);

    int gate_iters = (iters_A == iters_B);
    int gate_bytes = (bytes > 1000);  /* trace must be non-trivial */
    int gate_match = identical;

    printf("[determinism]   gate iter count        : run-A=%d run-B=%d -> %s\n",
           (int)iters_A, (int)iters_B, gate_iters ? "PASS" : "FAIL");
    printf("[determinism]   gate non-trivial trace : compared %ld bytes -> %s\n",
           bytes, gate_bytes ? "PASS" : "FAIL");
    printf("[determinism]   gate byte-identical    : first diff at byte %ld -> %s\n",
           first_diff, gate_match ? "PASS" : "FAIL");
    pass = (gate_iters && gate_bytes && gate_match) ? 1 : 0;
    printf("[determinism] %s\n", pass ? "[PASS]" : "[FAIL]");
  }
  PetscCallMPI(MPI_Bcast(&pass, 1, MPI_INT, 0, PETSC_COMM_WORLD));

  PetscCall(PetscFinalize());
  return pass ? 0 : 1;
}
