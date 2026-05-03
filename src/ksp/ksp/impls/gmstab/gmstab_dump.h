/*
   Diagnostic dump infrastructure for KSPGMSTAB.

   Header-only — see comment block at end for why. When the env var
   GMSTAB_DUMP_DIR is set to a directory path, each call to one of the
   dump functions emits a single CSV file (one row per matrix row,
   %.17g doubles). Filenames encode the checkpoint number + quantity
   name so they can be diff'ed pairwise against the matching files
   emitted by the C++ port (gmstab_cpp/include/gmstab/dump.hpp).

   Parallel correctness: the helpers transparently gather distributed
   PETSc Vec / Mat objects onto rank 0 and write a SINGLE consolidated
   file representing the global object. The file format is identical
   regardless of MPI rank count, so a sequential dump and a parallel
   dump of the same algebra produce comparable files (modulo the
   floating-point rounding that comes from differing reduction orders
   on different rank counts).

   When GMSTAB_DUMP_DIR is unset, every dump call is a no-op (cheap
   one-getenv check, no I/O, no MPI traffic). Production code is
   unaffected.

   File layout per quantity:
     <dump_dir>/petsc_chk<NN>_<name>.csv   — emitted by this PETSc port
     <dump_dir>/cpp_chk<NN>_<name>.csv     — emitted by the C++ port

   The Python diff tool (tests/diff_gmstab_dumps.py) walks the two
   prefixes and reports per-file Frobenius diff, max abs diff, and
   the first checkpoint where any quantity diverges.

   ----- Why header-only -----
   PETSc's build system builds its source-file manifest at configure time.
   Adding a new .c file under src/ksp/ksp/impls/gmstab/ would require
   re-running configure or manually editing the link manifest at the
   arch-XXX/lib/libpetsc.so.YY.args file. By keeping the helpers as
   static-inline functions in this header, every translation unit that
   includes the header gets its own private copies, which is fine
   because the helpers are tiny and only invoked when GMSTAB_DUMP_DIR is
   set (i.e. during debugging, never in production).
*/
#pragma once

#include <petsc/private/petscimpl.h>
#include <petscvec.h>
#include <petscmat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline PetscBool KSPGMSTABDumpEnabled_Private(void)
{
  const char *d = getenv("GMSTAB_DUMP_DIR");
  return (d && d[0]) ? PETSC_TRUE : PETSC_FALSE;
}

static inline PetscErrorCode KSPGMSTABDumpBuildPath_Private(PetscInt checkpoint, const char *name,
                                                             char *buf, size_t bufsz)
{
  PetscFunctionBegin;
  const char *d = getenv("GMSTAB_DUMP_DIR");
  PetscCheck(d && d[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "GMSTAB_DUMP_DIR unset");
  int rc = snprintf(buf, bufsz, "%s/petsc_chk%02" PetscInt_FMT "_%s.csv", d, checkpoint, name);
  PetscCheck(rc > 0 && (size_t)rc < bufsz, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "build_path: snprintf overflow");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------- Replicated-on-every-rank helpers (small-dense + scalars) ---------
   These dump quantities that are bit-identical on every rank: small-dense
   PetscScalar arrays computed via global reductions (VecDot, VecNorm,
   collective LAPACK-on-replicated-arrays) and scalars derived from them.
   Only rank 0 actually writes the file; other ranks no-op.
   ------------------------------------------------------------------------- */

static inline PetscErrorCode KSPGMSTABDumpDense_Private(PetscInt checkpoint, const char *name,
                                                         const PetscScalar *A, PetscInt ld,
                                                         PetscInt m, PetscInt n)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
  char path[1024];
  PetscCall(KSPGMSTABDumpBuildPath_Private(checkpoint, name, path, sizeof(path)));
  FILE *fp = fopen(path, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", path);
  for (PetscInt i = 0; i < m; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      const PetscScalar v = A[(size_t)i + (size_t)j * (size_t)ld];
      fprintf(fp, "%s%.17g", j ? "," : "", (double)PetscRealPart(v));
    }
    fputc('\n', fp);
  }
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSPGMSTABDumpReal_Private(PetscInt checkpoint, const char *name,
                                                        PetscReal value)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
  char path[1024];
  PetscCall(KSPGMSTABDumpBuildPath_Private(checkpoint, name, path, sizeof(path)));
  FILE *fp = fopen(path, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", path);
  fprintf(fp, "%.17g\n", (double)value);
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSPGMSTABDumpVector1d_Private(PetscInt checkpoint, const char *name,
                                                            const PetscScalar *v, PetscInt n)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);
  char path[1024];
  PetscCall(KSPGMSTABDumpBuildPath_Private(checkpoint, name, path, sizeof(path)));
  FILE *fp = fopen(path, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", path);
  for (PetscInt i = 0; i < n; ++i) fprintf(fp, "%.17g\n", (double)PetscRealPart(v[i]));
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* -------- Distributed Vec / Mat helpers ----------------------------------
   These gather the parallel object onto rank 0 via VecScatterCreateToZero
   (column-by-column for matrices) and have rank 0 write the consolidated
   global view. The output file is identical in structure regardless of
   rank count.
   ------------------------------------------------------------------------- */

static inline PetscErrorCode KSPGMSTABDumpVec_Private(PetscInt checkpoint, const char *name, Vec v)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);

  /* Determine rank in v's communicator. */
  PetscMPIInt rank, size;
  MPI_Comm comm = PetscObjectComm((PetscObject)v);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  Vec        v_seq    = NULL;
  VecScatter scat     = NULL;

  if (size > 1) {
    /* Gather all entries onto rank 0 in ascending global-index order. */
    PetscCall(VecScatterCreateToZero(v, &scat, &v_seq));
    PetscCall(VecScatterBegin(scat, v, v_seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(  scat, v, v_seq, INSERT_VALUES, SCATTER_FORWARD));
  }

  if (rank == 0) {
    PetscInt           N;
    const PetscScalar *arr;
    if (size > 1) {
      PetscCall(VecGetSize(v_seq, &N));
      PetscCall(VecGetArrayRead(v_seq, &arr));
    } else {
      PetscCall(VecGetSize(v, &N));
      PetscCall(VecGetArrayRead(v, &arr));
    }
    char path[1024];
    PetscCall(KSPGMSTABDumpBuildPath_Private(checkpoint, name, path, sizeof(path)));
    FILE *fp = fopen(path, "w");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", path);
    for (PetscInt i = 0; i < N; ++i) fprintf(fp, "%.17g\n", (double)PetscRealPart(arr[i]));
    fclose(fp);
    if (size > 1) PetscCall(VecRestoreArrayRead(v_seq, &arr));
    else          PetscCall(VecRestoreArrayRead(v, &arr));
  }

  if (size > 1) {
    PetscCall(VecScatterDestroy(&scat));
    PetscCall(VecDestroy(&v_seq));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSPGMSTABDumpMatCols_Private(PetscInt checkpoint, const char *name,
                                                            Mat M, PetscInt ncols);

static inline PetscErrorCode KSPGMSTABDumpMat_Private(PetscInt checkpoint, const char *name, Mat M)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt n;
  PetscCall(MatGetSize(M, NULL, &n));
  PetscCall(KSPGMSTABDumpMatCols_Private(checkpoint, name, M, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Dump only the first `ncols` columns of M. Required when the underlying
   storage is over-allocated (e.g., ws->W has m_max+1 columns but cycle1
   only uses s+1). Calling KSPGMSTABDumpMat_Private would gather the full
   m_max+1 columns; this variant lets the caller restrict to the active
   slice while still doing a proper parallel gather (which must NOT be
   confused with the small-dense path: ws->W is distributed, not
   replicated). */
static inline PetscErrorCode KSPGMSTABDumpMatCols_Private(PetscInt checkpoint, const char *name,
                                                            Mat M, PetscInt ncols)
{
  PetscFunctionBegin;
  if (!KSPGMSTABDumpEnabled_Private()) PetscFunctionReturn(PETSC_SUCCESS);

  PetscMPIInt rank, size;
  MPI_Comm comm = PetscObjectComm((PetscObject)M);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscInt m, n_full;
  PetscCall(MatGetSize(M, &m, &n_full));
  PetscCheck(ncols >= 0 && ncols <= n_full, comm, PETSC_ERR_ARG_OUTOFRANGE,
             "DumpMatCols: ncols=%" PetscInt_FMT " out of [0, %" PetscInt_FMT "]", ncols, n_full);
  const PetscInt n = ncols;

  /* Rank 0 collects the m×n matrix in column-major as it scatters each
     column. Other ranks just participate in the scatters. */
  PetscScalar *gathered = NULL;
  if (rank == 0) PetscCall(PetscMalloc1((size_t)m * (size_t)n, &gathered));

  if (size == 1) {
    /* Sequential fast path: copy the dense array directly. */
    if (rank == 0) {
      const PetscScalar *arr;
      PetscInt           ld;
      PetscCall(MatDenseGetArrayRead(M, &arr));
      PetscCall(MatDenseGetLDA(M, &ld));
      for (PetscInt j = 0; j < n; ++j)
        for (PetscInt i = 0; i < m; ++i)
          gathered[(size_t)i + (size_t)j * (size_t)m] =
              arr[(size_t)i + (size_t)j * (size_t)ld];
      PetscCall(MatDenseRestoreArrayRead(M, &arr));
    }
  } else {
    /* Parallel path: scatter each column onto rank 0. */
    for (PetscInt j = 0; j < n; ++j) {
      Vec        col, col_seq = NULL;
      VecScatter scat;
      PetscCall(MatDenseGetColumnVecRead(M, j, &col));
      PetscCall(VecScatterCreateToZero(col, &scat, &col_seq));
      PetscCall(VecScatterBegin(scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(  scat, col, col_seq, INSERT_VALUES, SCATTER_FORWARD));
      if (rank == 0) {
        const PetscScalar *carr;
        PetscCall(VecGetArrayRead(col_seq, &carr));
        for (PetscInt i = 0; i < m; ++i)
          gathered[(size_t)i + (size_t)j * (size_t)m] = carr[i];
        PetscCall(VecRestoreArrayRead(col_seq, &carr));
      }
      PetscCall(VecScatterDestroy(&scat));
      PetscCall(VecDestroy(&col_seq));
      PetscCall(MatDenseRestoreColumnVecRead(M, j, &col));
    }
  }

  if (rank == 0) {
    char path[1024];
    PetscCall(KSPGMSTABDumpBuildPath_Private(checkpoint, name, path, sizeof(path)));
    FILE *fp = fopen(path, "w");
    PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open %s for writing", path);
    for (PetscInt i = 0; i < m; ++i) {
      for (PetscInt j = 0; j < n; ++j) {
        const PetscScalar v = gathered[(size_t)i + (size_t)j * (size_t)m];
        fprintf(fp, "%s%.17g", j ? "," : "", (double)PetscRealPart(v));
      }
      fputc('\n', fp);
    }
    fclose(fp);
    PetscCall(PetscFree(gathered));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
