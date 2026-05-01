# Bug report — `PCAMGX` cannot consume MATAIJ matrices that advertise `bSize > 1`

**Component:** `src/ksp/pc/impls/amgx/amgx.cxx`
**Severity:** High — fundamental functionality. Block-structured AIJ matrices
(e.g. those produced by `DMDA` with `ndof > 1`, or any `MatSetBlockSize(>1)`
on a `MATAIJ`) are unusable with PCAMGX. This is the natural data type for
multi-component PDE codes (CFD, MHD, HDG / DG), which is the precise
audience PCAMGX is meant to serve.
**PETSc version:** Development branch, `v3.25.0-171-g7d1cbee15b2` (2026-04-30).
**AMGX version:** 2.4.0 (per `--download-amgx`).

---

## Summary

When the matrix passed to PCAMGX is `MATAIJ` (or `MATAIJCUSPARSE`) with an
advertised block size `bSize > 1`, `PCSetUp_AMGX` calls
`AMGX_matrix_upload_distributed` with **scalar-coordinate** row offsets,
column indices, partition offsets, and values, but with **block dimensions**
`(bSize, bSize)`. AMGX's contract for `bSize > 1` requires
**block-coordinate** indices and **block-major** values; passing scalar
data with `bSize > 1` causes AMGX to walk past the end of the values
buffer.

Observed manifestations (all reproducible on `MATAIJCUSPARSE` with
`MatSetBlockSize(A, 4)` and a representative matrix):

- A confused mix of host (rowOff/colIdx) and device (values) pointers given
  to AMGX → **hang** inside `cuMemcpy`-style copies (single-rank).
- AMGX `FatalError("memcpy of size_t(-1) bytes")` (some matrix sizes).
- Garbage / unsymmetric divergence when AMGX silently misreads the values.

This shut out a broad class of natural usage. The most common production
example: `src/snes/tutorials/ex19.c` (lid-driven cavity, DMDA `ndof = 4`),
which is the standard PETSc demonstration of multi-component CFD and is
the matrix type our HDG / DG production codes also produce.

---

## Reproducer

`src/snes/tutorials/ex19.c` with PCAMGX:

```bash
mpirun -n 1 ./ex19 \
    -da_grid_x 8 -da_grid_y 8 \
    -mat_type aijcusparse -vec_type cuda \
    -ksp_type gmres \
    -pc_type amgx \
    -pc_amgx_amg_method CLASSICAL \
    -pc_amgx_smoother JACOBI_L1 \
    -pc_amgx_selector PMIS
```

Without the fix, AMGX hangs in setup; with the fix, the matrix uploads
correctly. (The downstream solve in this exact configuration is
constrained by AMGX 2.4.0's CLASSICAL-AMG handling of block matrices,
which is a separate AMGX-side limitation; with
`-pc_amgx_amg_method AGGREGATION -pc_amgx_smoother BLOCK_JACOBI
-pc_amgx_selector SIZE_4 -pc_amgx_max_levels 1
-pc_amgx_coarse_solver NOSOLVER`, ex19 converges end-to-end.)

A more focused, self-contained reproducer is in
`test_amgx_block_aij.cpp`: it constructs a small block-tridiagonal SPD
problem at four block sizes (`bSize ∈ {1, 2, 4, 9}`), routes it through
PCAMGX via `KSPSetUp` + `KSPSolve`, and verifies that:

- `bSize == 1` continues to converge (regression).
- `bSize > 1` survives `KSPSetUp` cleanly (the load-bearing change).
- `bSize == 4` converges end-to-end on a single rank.
- `MATSEQBAIJ` is rejected with a friendly `PETSC_ERR_SUP` and a
  `MatConvert(MATAIJ)` directive (rather than a cryptic crash inside
  `MatSeqAIJGetArrayRead`).

---

## Root cause

`PCSetUp_AMGX` (pre-fix) at `amgx.cxx:296`:

```cpp
PetscCallAmgX(AMGX_matrix_upload_distributed(
    amgx->A,
    amgx->nGlobalRows,        // scalar global rows
    (int)amgx->nLocalRows,    // scalar local rows
    (int)amgx->nnz,           // scalar local nnz
    amgx->bSize,              // bDimX (correctly the block size)
    amgx->bSize,              // bDimY
    rowOffsets,               // scalar CSR row offsets
    colIndices,               // scalar CSR col indices
    amgx->values,             // scalar values
    NULL, dist));
```

AMGX's `AMGX_matrix_upload_distributed`, when `bDimX = bDimY = bSize > 1`,
expects:

| parameter | scalar coords | block coords (bSize > 1) |
|---|---|---|
| `n` (global rows) | total scalar rows | global block rows = `globalRows / bSize` |
| `nrows` (local rows) | total scalar rows | local block rows = `localRows / bSize` |
| `nnz` (local nnz) | scalar nnz | block nnz = `scalarNnz / (bSize * bSize)` |
| `row_ptrs` | length `nrows + 1`, offsets into `col_indices` | same shape, but everything in block coords |
| `col_indices` | scalar column indices | block column indices |
| `values` | length `nnz` | length `nnz * bSize * bSize`, block-major (row-major within block) |
| partition offsets (via `AMGX_distribution_set_partition_data`) | scalar | block coords |

The pre-fix code passed scalar-coord data with `bDimX = bDimY = bSize`, so
AMGX interpreted the scalar buffers as block buffers and walked off the
end of `values`. Hence the hang / size_t(-1) memcpy / garbage outputs.

`PCApply_AMGX` had a parallel issue: `AMGX_vector_upload(handle, n,
block_dim, data)` was called with `n = nLocalRows` and `block_dim = 1`,
giving `n_blocks * block_dim = nLocalRows * 1 = nLocalRows` total scalars
— which is correct in count but disagrees with the matrix's stated block
shape (the bound matrix has `bSize > 1`, so AMGX expects
`n_blocks * block_dim` to come from `n_blocks = nLocalRows / bSize` with
`block_dim = bSize`).

---

## Why we did not require MATBAIJ

There is a tempting alternative fix: require users to pass a
`MATSEQBAIJ` / `MATMPIBAIJ` matrix (whose internal storage is already
block-coordinate). It looks simpler and avoids a host-side repack. We
rejected it for three reasons:

1. **There is no `MATAIJCUSPARSE`-equivalent BAIJ on the GPU in PETSc
   today.** Requiring BAIJ would force users to choose between PCAMGX
   (which lives on the GPU) and a GPU-resident matrix (which is the whole
   point of PCAMGX). That is not a real choice.
2. **PETSc's other block-aware preconditioners (PCGAMG, PCHYPRE,
   PCPBJACOBI, etc.) all accept MATAIJ + `MatSetBlockSize > 1`.** Forcing
   PCAMGX alone to require BAIJ would be an inconsistent surface and
   would drive the same down-stream conversion that this fix performs
   internally — only without a clear error message when users
   forget.
3. **`DMDA` produces `MATAIJ` with bSize metadata, not BAIJ.** This is the
   canonical block-matrix factory in PETSc; users build matrices via
   `DMCreateMatrix(DMDA(ndof=4), ...)` and then expect to hand the
   result to any PC. Requiring BAIJ would break this.

Given these constraints, the right boundary is: PCAMGX accepts the
PETSc-natural block-AIJ surface, and translates internally to AMGX's
demanded block-coordinate format.

---

## Fix overview

`PCSetUp_AMGX` now branches on `amgx->bSize`:

- **`bSize == 1`:** unchanged scalar-CSR upload (verified via the
  `bSize == 1` regression test).
- **`bSize > 1`:** repack the scalar AIJ into AMGX-compatible block CSR on
  the host:
  - Pass 1: per block row, gather the *union* of touched block columns
    across the `bSize` composing scalar rows (sort + unique).
  - Pass 2: prefix-sum into block row offsets. Validate that block nnz
    fits in 32-bit (AMGX's index restriction).
  - Pass 3: flatten into block-coordinate column indices, zero-init the
    block-major values buffer (size `nnzBlocks * bSize * bSize`).
  - Pass 4: scatter scalar entries into their block-major positions
    (`values[k * bSize * bSize + r * bSize + c]`).
  - Densification: missing intra-block scalar entries become explicit
    zeros, matching how other PETSc-AMGX-style bindings (notably PCHYPRE)
    feed BAIJ-flavoured solvers from MATAIJ inputs.
  - Partition offsets are translated to block coordinates and validated
    to be `bSize`-aligned (since block rows cannot straddle ranks).

`PCApply_AMGX` is updated to pass `n = nLocalRows / bSize`, `block_dim =
bSize` to both vector uploads, matching the matrix's block shape. For
`bSize == 1`, this reduces to the previous form (no behavior change).

`MATBAIJ` inputs are explicitly rejected with `PETSC_ERR_SUP` and a clear
remediation: `MatConvert(A, MATAIJ, MAT_INPLACE_MATRIX, &A);` preserves
the bSize metadata, and the AIJ + bSize > 1 path then handles the matrix
correctly.

A small adjacent fix piggy-backs in: `PCSetFromOptions_AMGX`'s selector /
AMG-method consistency check used the *cached* (previous) selector enum
rather than the *just-parsed* one, so switching from `CLASSICAL` (default
selector `PMIS`) to `AGGREGATION` plus an explicit selector like `SIZE_2`
incorrectly errored. The check now uses the just-parsed selector (and the
error message lists the valid options).

---

## Testing

| test | scenario | result |
|---|---|---|
| `test_amgx_block_aij` | `MATAIJ`, `bSize=1`, scalar (CPU) + GPU. CLASSICAL+JACOBI_L1+PMIS, AMGX V-cycle. | converges in 16 iters (slow + fast paths) — no regression |
| `test_amgx_block_aij` | `MATAIJCUSPARSE`, `bSize=2`. AGGREGATION+BLOCK_JACOBI+SIZE_2, max_levels=1. | `KSPSetUp` clean. Solve hits an AMGX-internal `non-distributed matrix` check inside `BlockJacobiSolver::smooth_BxB → bsrmv` — separate AMGX 2.4.0 limitation, not a binding bug. |
| `test_amgx_block_aij` | `MATAIJCUSPARSE`, `bSize=4`. AGGREGATION+BLOCK_JACOBI+SIZE_2, max_levels=1. | converges end-to-end in 8 iters (slow + fast paths) ✅ |
| `test_amgx_block_aij` | `MATAIJCUSPARSE`, `bSize=9` (HDG production target). | `KSPSetUp` clean. Solve hits the same AMGX-internal limitation as `bSize=2`. |
| `test_amgx_block_aij` | `MATSEQBAIJ`, `bSize=4`. | rejected with `PETSC_ERR_SUP` and a clean remediation message; subsequent `KSPDestroy` succeeds (relies on the destructor fix at `2d93faec18c`). |
| `test_amgx_destroy_recovery` | the partial-init recovery test from `2d93faec18c` | continues to pass (no regression). |
| `src/snes/tutorials/ex19` | `DMDA(ndof=4)` lid-driven cavity, MATAIJCUSPARSE, AGGREGATION+BLOCK_JACOBI+SIZE_4+max_levels=1+NOSOLVER. | SNES converges in 2 outer iters; KSP converges in 1 inner per outer ✅. Pre-fix this hung in setup; post-fix it produces the correct solution. |

The full test suite is gated by:

- `test_amgx_block_aij.cpp` — host-only build instructions in the file
  header; run the binary, exit 0 ⇒ all scenarios passed.
- `test_amgx_destroy_recovery.cpp` — host-only sanity test; exit 0 ⇒
  destructor recovery still works.
- `ex19` invocation above — quick end-to-end DMDA bs=4 sanity.

---

## Two AMGX 2.4.0 caveats observed during testing (not binding bugs)

These showed up while validating the fix and are worth recording in case
future PCAMGX users hit them:

1. **AGGREGATION + SIZE_x selector forces device→host migration of coarse
   levels.** AMGX's defaults `min_fine_rows=1 < min_coarse_rows=2` cause
   `AMG_Setup` to allocate a host-resident coarsest level. The
   `SIZE_2/4/8/MULTI_PAIRWISE` selectors throw "setAggregates not
   implemented on CPU" on that host level. Workarounds: set
   `min_fine_rows == min_coarse_rows` (no PETSc binding option for this
   today), or use `max_levels = 1` (degenerates to block-Jacobi
   smoothing on the input matrix), or force everything onto a single
   level via `coarse_solver = NOSOLVER`.

2. **`BlockJacobiSolver::smooth_BxB` requires a "distributed matrix"
   even on `MPI_COMM_SELF` for some block sizes.** The `bsrmv` path goes
   through `getFixedSizesForView` which fatal-errors on
   non-distributed matrices. This affects the single-rank single-process
   block-AGG configuration for some block sizes (we observed `bSize=2,9`
   fail, `bSize=4` succeed). The production HDG / DG use case is
   multi-rank distributed, so this does not affect deployed usage; it
   only complicates writing self-contained regression tests on a single
   rank.

Both could be addressed independently, but neither is in scope for this
fix; the binding bug is purely on the PETSc side and is fixed
unconditionally.

---

## Effort estimate

- **Code change (binding):** ~150 lines in `amgx.cxx`. Took ~4 hours
  including diagnostic injection, root-cause confirmation, and final
  cleanup.
- **Verification:** the regression test (`test_amgx_block_aij.cpp`,
  ~330 LOC) plus an `ex19` invocation. Took ~3 hours including
  configuration sweeps to disentangle binding bugs from AMGX-internal
  limitations.
- **Review:** standard PETSc workflow.

---

## Submitter notes

- Discovered while extending the comparison harness from the destructor
  bug report (`AMGX_DESTROY_BUG.md` / commit `2d93faec18c`) to cover
  multi-component HDG matrices (`bSize = 9`).
- The HDG production target is `N ≈ 4.7M`, `nnz ≈ 213M`, `bSize = 9`
  on a single V100, multi-rank. Single-process testing was used here
  for tractable reproducer footprint.
- Full reproducer + standalone test source can be made available;
  contact via the channel of submission.
