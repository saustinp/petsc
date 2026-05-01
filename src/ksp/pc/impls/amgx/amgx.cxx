/*
     This file implements an AmgX preconditioner in PETSc as part of PC.
 */

/*
   Include files needed for the AmgX preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <petscdevice_cuda.h>
#include <amgx_c.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>

enum class AmgXSmoother {
  PCG,
  PCGF,
  PBiCGStab,
  GMRES,
  FGMRES,
  JacobiL1,
  BlockJacobi,
  GS,
  MulticolorGS,
  MulticolorILU,
  MulticolorDILU,
  ChebyshevPoly,
  NoSolver
};
enum class AmgXAMGMethod {
  Classical,
  Aggregation
};
enum class AmgXSelector {
  Size2,
  Size4,
  Size8,
  MultiPairwise,
  PMIS,
  HMIS
};
enum class AmgXCoarseSolver {
  DenseLU,
  NoSolver
};
enum class AmgXAMGCycle {
  V,
  W,
  F,
  CG,
  CGF
};

struct AmgXControlMap {
  static const std::map<std::string, AmgXAMGMethod>    AMGMethods;
  static const std::map<std::string, AmgXSmoother>     Smoothers;
  static const std::map<std::string, AmgXSelector>     Selectors;
  static const std::map<std::string, AmgXCoarseSolver> CoarseSolvers;
  static const std::map<std::string, AmgXAMGCycle>     AMGCycles;
};

const std::map<std::string, AmgXAMGMethod> AmgXControlMap::AMGMethods = {
  {"CLASSICAL",   AmgXAMGMethod::Classical  },
  {"AGGREGATION", AmgXAMGMethod::Aggregation}
};

const std::map<std::string, AmgXSmoother> AmgXControlMap::Smoothers = {
  {"PCG",             AmgXSmoother::PCG           },
  {"PCGF",            AmgXSmoother::PCGF          },
  {"PBICGSTAB",       AmgXSmoother::PBiCGStab     },
  {"GMRES",           AmgXSmoother::GMRES         },
  {"FGMRES",          AmgXSmoother::FGMRES        },
  {"JACOBI_L1",       AmgXSmoother::JacobiL1      },
  {"BLOCK_JACOBI",    AmgXSmoother::BlockJacobi   },
  {"GS",              AmgXSmoother::GS            },
  {"MULTICOLOR_GS",   AmgXSmoother::MulticolorGS  },
  {"MULTICOLOR_ILU",  AmgXSmoother::MulticolorILU },
  {"MULTICOLOR_DILU", AmgXSmoother::MulticolorDILU},
  {"CHEBYSHEV_POLY",  AmgXSmoother::ChebyshevPoly },
  {"NOSOLVER",        AmgXSmoother::NoSolver      }
};

const std::map<std::string, AmgXSelector> AmgXControlMap::Selectors = {
  {"SIZE_2",         AmgXSelector::Size2        },
  {"SIZE_4",         AmgXSelector::Size4        },
  {"SIZE_8",         AmgXSelector::Size8        },
  {"MULTI_PAIRWISE", AmgXSelector::MultiPairwise},
  {"PMIS",           AmgXSelector::PMIS         },
  {"HMIS",           AmgXSelector::HMIS         }
};

const std::map<std::string, AmgXCoarseSolver> AmgXControlMap::CoarseSolvers = {
  {"DENSE_LU_SOLVER", AmgXCoarseSolver::DenseLU },
  {"NOSOLVER",        AmgXCoarseSolver::NoSolver}
};

const std::map<std::string, AmgXAMGCycle> AmgXControlMap::AMGCycles = {
  {"V",   AmgXAMGCycle::V  },
  {"W",   AmgXAMGCycle::W  },
  {"F",   AmgXAMGCycle::F  },
  {"CG",  AmgXAMGCycle::CG },
  {"CGF", AmgXAMGCycle::CGF}
};

/*
   Private context (data structure) for the AMGX preconditioner.
*/
struct PC_AMGX {
  AMGX_solver_handle    solver;
  AMGX_config_handle    cfg;
  AMGX_resources_handle rsrc;
  bool                  solve_state_init;
  bool                  rsrc_init;
  PetscBool             verbose;

  AMGX_matrix_handle A;
  AMGX_vector_handle sol;
  AMGX_vector_handle rhs;

  MPI_Comm    comm;
  PetscMPIInt rank   = 0;
  PetscMPIInt nranks = 0;
  int         devID  = 0;

  void       *lib_handle = 0;
  std::string cfg_contents;

  // Cached state for re-setup. Counts here are SCALAR-coordinate quantities
  // (i.e. matching the underlying MATAIJ's row/column space). When the
  // matrix advertises a block size > 1, AMGX's matrix and replace-coefficient
  // calls take BLOCK-coordinate counts derived from these (nLocalRows / bSize,
  // and nnzBlocks). nnzBlocks is populated by PCSetUp_AMGX on the slow path
  // and re-validated on the fast path; for bSize == 1 it equals nnz.
  PetscInt           nnz;
  PetscInt           nnzBlocks;
  PetscInt           nLocalRows;
  PetscInt           nGlobalRows;
  PetscInt           bSize;
  Mat                localA;
  const PetscScalar *values;

  // AMG Control parameters
  AmgXSmoother     smoother;
  AmgXAMGMethod    amg_method;
  AmgXSelector     selector;
  AmgXCoarseSolver coarse_solver;
  AmgXAMGCycle     amg_cycle;
  PetscInt         presweeps;
  PetscInt         postsweeps;
  PetscInt         max_levels;
  PetscInt         aggressive_levels;
  PetscInt         dense_lu_num_rows;
  PetscScalar      strength_threshold;
  PetscBool        print_grid_stats;
  PetscBool        exact_coarse_solve;

  // Smoother control parameters
  PetscScalar jacobi_relaxation_factor;
  PetscScalar gs_symmetric;
};

static PetscInt s_count = 0;

// Buffer of messages from AmgX
// Currently necessary hack before we adapt AmgX to print from single rank only
static std::string amgx_output{};

// A print callback that allows AmgX to return status messages
static void print_callback(const char *msg, int length)
{
  amgx_output.append(msg);
}

// Outputs messages from the AmgX message buffer and clears it
static PetscErrorCode amgx_output_messages(PC_AMGX *amgx)
{
  PetscFunctionBegin;
  // If AmgX output is enabled and we have a message, output it
  if (amgx->verbose && !amgx_output.empty()) {
    // Only a single rank to output the AmgX messages
    PetscCall(PetscPrintf(amgx->comm, "AMGX: %s", amgx_output.c_str()));

    // Note that all ranks clear their received output
    amgx_output.clear();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// XXX Need to add call in AmgX API that gracefully destroys everything
// without abort etc.
#define PetscCallAmgX(rc) \
  do { \
    AMGX_RC err = (rc); \
    char    msg[4096]; \
    switch (err) { \
    case AMGX_RC_OK: \
      break; \
    default: \
      AMGX_get_error_string(err, msg, 4096); \
      SETERRQ(amgx->comm, PETSC_ERR_LIB, "%s", msg); \
    } \
  } while (0)

/*
   PCAMGXBuildBlockCSR_Private - Repack a host-resident scalar AIJ CSR matrix
   into block-coordinate CSR with block-major (row-major within block) values,
   suitable for AMGX_matrix_upload_distributed when bSize > 1.

   AMGX expects, when blockDimX = blockDimY = bSize > 1:
     - row offsets and column indices in BLOCK coordinates (length
       nBlockRows+1 and nnzBlocks respectively, where nBlockRows = nScalarRows
       / bSize)
     - values stored block-major: for the k-th block,
         values[k * bSize * bSize + r * bSize + c]  is the entry at
         (block-row, block-col) (k -> iBlk, jBlk(k)) at intra-block position
         (r, c). Within each block, layout is row-major.

   PETSc MATAIJ stores scalar CSR. When the user advertises bSize > 1 (via
   MatSetBlockSize, or implicitly via DMDA with ndof > 1), the scalar CSR may
   not contain every (r, c) entry inside each block — some intra-block entries
   may be structurally absent. We "densify" each block: for every (iBlk, jBlk)
   block touched by any of the bSize scalar rows, we materialize the full
   bSize * bSize block, padding missing scalar entries with 0.0. This matches
   how other PETSc bindings (e.g. PCHYPRE) hand block matrices to libraries
   that demand full BAIJ-style storage.

   The function does the repack on the host. For matrices whose values live
   on the device, the caller must copy them to a host buffer first.

   Inputs:
     bSize        - block size (>= 2; caller handles bSize == 1 separately)
     nScalarRows  - local rows in the scalar AIJ representation
                    (must be divisible by bSize)
     scalarRowOff - host array, length nScalarRows + 1
     scalarColIdx - host array, length scalarRowOff[nScalarRows]
     scalarValues - host array, length scalarRowOff[nScalarRows]

   Outputs (written into the std::vector& parameters; caller need not pre-size):
     blockRowOff  - length nBlockRows + 1; AMGX-compatible int
     blockColIdx  - length nnzBlocks
     blockValues  - length nnzBlocks * bSize * bSize, block-major / row-major
     nnzBlocksOut - number of block nonzeros (also implicit in
                    blockRowOff.back())
*/
static PetscErrorCode PCAMGXBuildBlockCSR_Private(PetscInt bSize,
                                                  PetscInt nScalarRows,
                                                  const PetscInt    *scalarRowOff,
                                                  const PetscInt    *scalarColIdx,
                                                  const PetscScalar *scalarValues,
                                                  std::vector<int>         &blockRowOff,
                                                  std::vector<int>         &blockColIdx,
                                                  std::vector<PetscScalar> &blockValues,
                                                  PetscInt                 *nnzBlocksOut)
{
  PetscFunctionBegin;
  PetscCheck(bSize >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "PCAMGX repack: bSize=%" PetscInt_FMT " is invalid (must be >= 1)", bSize);
  PetscCheck(nScalarRows % bSize == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "PCAMGX repack: scalar local rows %" PetscInt_FMT
             " is not divisible by bSize=%" PetscInt_FMT, nScalarRows, bSize);

  const PetscInt nBlockRows = nScalarRows / bSize;

  /* Pass 1: for each block row, gather the union of touched block columns
     across the bSize composing scalar rows. We collect into a per-row
     scratch vector then sort/unique, rather than std::set, to limit
     allocator pressure for typical FE/HDG matrices where each block row
     has ~10s of block neighbors. */
  std::vector<std::vector<int>> blockColsPerRow((size_t)nBlockRows);
  for (PetscInt iBlk = 0; iBlk < nBlockRows; ++iBlk) {
    std::vector<int> &cols = blockColsPerRow[(size_t)iBlk];
    /* upper bound on size: total scalar nnz across the bSize scalar rows */
    PetscInt budget = 0;
    for (PetscInt r = 0; r < bSize; ++r) {
      const PetscInt iScalar = iBlk * bSize + r;
      budget += scalarRowOff[iScalar + 1] - scalarRowOff[iScalar];
    }
    cols.reserve((size_t)budget);
    for (PetscInt r = 0; r < bSize; ++r) {
      const PetscInt iScalar = iBlk * bSize + r;
      const PetscInt rs      = scalarRowOff[iScalar];
      const PetscInt re      = scalarRowOff[iScalar + 1];
      for (PetscInt k = rs; k < re; ++k) {
        cols.push_back((int)(scalarColIdx[k] / bSize));
      }
    }
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
  }

  /* Pass 2: prefix sum of per-row sizes to get blockRowOff. */
  blockRowOff.assign((size_t)(nBlockRows + 1), 0);
  for (PetscInt iBlk = 0; iBlk < nBlockRows; ++iBlk) {
    blockRowOff[(size_t)(iBlk + 1)] =
      blockRowOff[(size_t)iBlk] + (int)blockColsPerRow[(size_t)iBlk].size();
  }
  const PetscInt nnzBlocks = (PetscInt)blockRowOff[(size_t)nBlockRows];
  PetscCheck(nnzBlocks <= std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "PCAMGX repack: block nnz %" PetscInt_FMT " exceeds int range; "
             "AMGX restricts block nnz to 32-bit indices.", nnzBlocks);

  /* Pass 3: flatten per-row block-column lists into blockColIdx, init values to 0. */
  blockColIdx.resize((size_t)nnzBlocks);
  for (PetscInt iBlk = 0; iBlk < nBlockRows; ++iBlk) {
    const std::vector<int> &cols = blockColsPerRow[(size_t)iBlk];
    std::copy(cols.begin(), cols.end(),
              blockColIdx.begin() + blockRowOff[(size_t)iBlk]);
  }
  blockValues.assign((size_t)nnzBlocks * (size_t)bSize * (size_t)bSize, (PetscScalar)0.0);

  /* Pass 4: scatter scalar entries into block-major positions.
     For each scalar (iScalar, jScalar, value), find its containing block
     (iBlk, jBlk) and intra-block position (r, c). Locate jBlk's slot via
     binary search within the block row's sorted block-col list. */
  for (PetscInt iBlk = 0; iBlk < nBlockRows; ++iBlk) {
    const std::vector<int> &cols   = blockColsPerRow[(size_t)iBlk];
    const PetscInt          kStart = blockRowOff[(size_t)iBlk];
    for (PetscInt r = 0; r < bSize; ++r) {
      const PetscInt iScalar = iBlk * bSize + r;
      const PetscInt rs      = scalarRowOff[iScalar];
      const PetscInt re      = scalarRowOff[iScalar + 1];
      for (PetscInt k = rs; k < re; ++k) {
        const PetscInt jScalar = scalarColIdx[k];
        const int      jBlk    = (int)(jScalar / bSize);
        const PetscInt c       = jScalar - (PetscInt)jBlk * bSize;
        auto it = std::lower_bound(cols.begin(), cols.end(), jBlk);
        PetscCheck(it != cols.end() && *it == jBlk, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                   "PCAMGX repack: block-col index lookup failed (iBlk=%" PetscInt_FMT
                   ", jBlk=%d) — internal invariant broken.", iBlk, jBlk);
        const PetscInt kBlk = kStart + (PetscInt)(it - cols.begin());
        blockValues[(size_t)kBlk * (size_t)bSize * (size_t)bSize
                    + (size_t)r * (size_t)bSize + (size_t)c] = scalarValues[k];
      }
    }
  }

  *nnzBlocksOut = nnzBlocks;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCSetUp_AMGX - Prepares for the use of the AmgX preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Note:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
static PetscErrorCode PCSetUp_AMGX(PC pc)
{
  PC_AMGX  *amgx = (PC_AMGX *)pc->data;
  Mat       Pmat = pc->pmat;
  PetscBool is_dev_ptrs;
  PetscBool is_baij;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)Pmat, &is_dev_ptrs, MATAIJCUSPARSE, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
  /* PCAMGX uses scalar AIJ as its canonical input form. MATBAIJ is rejected
     here with a directive to convert; this preserves the bSize metadata via
     MatConvert and lets the AIJ + bSize > 1 path below handle the block
     repack. (There is also no GPU BAIJ in PETSc, so requiring BAIJ would
     defeat the GPU AMGX path entirely.) */
  PetscCall(PetscObjectTypeCompareAny((PetscObject)Pmat, &is_baij, MATBAIJ, MATSEQBAIJ, MATMPIBAIJ, ""));
  PetscCheck(!is_baij, PetscObjectComm((PetscObject)Pmat), PETSC_ERR_SUP,
             "PCAMGX does not accept MATBAIJ matrices directly. Convert to MATAIJ "
             "(block-size metadata is preserved) via "
             "MatConvert(A, MATAIJ, MAT_INPLACE_MATRIX, &A); PCAMGX will repack "
             "the scalar AIJ into AMGX's block layout internally.");

  // At the present time, an AmgX matrix is a sequential matrix
  // Non-sequential/MPI matrices must be adapted to extract the local matrix
  bool partial_setup_allowed = (pc->setupcalled && pc->flag != DIFFERENT_NONZERO_PATTERN);
  if (amgx->nranks > 1) {
    if (partial_setup_allowed) {
      PetscCall(MatMPIAIJGetLocalMat(Pmat, MAT_REUSE_MATRIX, &amgx->localA));
    } else {
      PetscCall(MatMPIAIJGetLocalMat(Pmat, MAT_INITIAL_MATRIX, &amgx->localA));
    }

    if (is_dev_ptrs) PetscCall(MatConvert(amgx->localA, MATSEQAIJCUSPARSE, MAT_INPLACE_MATRIX, &amgx->localA));
  } else {
    amgx->localA = Pmat;
  }

  if (is_dev_ptrs) {
    PetscCall(MatSeqAIJCUSPARSEGetArrayRead(amgx->localA, &amgx->values));
  } else {
    PetscCall(MatSeqAIJGetArrayRead(amgx->localA, &amgx->values));
  }

  if (!partial_setup_allowed) {
    // Initialise resources and matrices
    if (!amgx->rsrc_init) {
      // Read configuration file
      PetscCallAmgX(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));
      PetscCallAmgX(AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID));
      amgx->rsrc_init = true;
    }

    PetscCheck(!amgx->solve_state_init, amgx->comm, PETSC_ERR_PLIB, "AmgX solve state initialisation already called.");
    PetscCallAmgX(AMGX_matrix_create(&amgx->A, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_vector_create(&amgx->sol, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_vector_create(&amgx->rhs, amgx->rsrc, AMGX_mode_dDDI));
    PetscCallAmgX(AMGX_solver_create(&amgx->solver, amgx->rsrc, AMGX_mode_dDDI, amgx->cfg));
    amgx->solve_state_init = true;

    // Extract the (scalar) CSR data. MatGetRowIJ returns scalar coordinates
    // for MATAIJ regardless of advertised block size; we re-pack into
    // block-coordinate CSR below when bSize > 1.
    PetscBool       done;
    const PetscInt *colIndices;
    const PetscInt *rowOffsets;
    PetscCall(MatGetRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &amgx->nLocalRows, &rowOffsets, &colIndices, &done));
    PetscCheck(done, amgx->comm, PETSC_ERR_PLIB, "MatGetRowIJ was not successful");
    PetscCheck(amgx->nLocalRows < std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "AmgX restricted to int local rows but nLocalRows = %" PetscInt_FMT " > max<int>", amgx->nLocalRows);

    if (is_dev_ptrs) {
      PetscCallCUDA(cudaMemcpy(&amgx->nnz, &rowOffsets[amgx->nLocalRows], sizeof(int), cudaMemcpyDefault));
    } else {
      amgx->nnz = rowOffsets[amgx->nLocalRows];
    }

    PetscCheck(amgx->nnz < std::numeric_limits<int>::max(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Support for 64-bit integer nnz not yet implemented, nnz = %" PetscInt_FMT ".", amgx->nnz);

    // Allocate space for some partition offsets (in scalar coords).
    std::vector<PetscInt> partitionOffsets(amgx->nranks + 1);

    // Fetch the number of local rows per rank
    partitionOffsets[0] = 0; /* could use PetscLayoutGetRanges */
    PetscCallMPI(MPI_Allgather(&amgx->nLocalRows, 1, MPIU_INT, partitionOffsets.data() + 1, 1, MPIU_INT, amgx->comm));
    std::partial_sum(partitionOffsets.begin(), partitionOffsets.end(), partitionOffsets.begin());

    // Fetch the number of global rows (scalar)
    amgx->nGlobalRows = partitionOffsets[amgx->nranks];

    PetscCall(MatGetBlockSize(Pmat, &amgx->bSize));
    PetscCheck(amgx->bSize >= 1, amgx->comm, PETSC_ERR_PLIB,
               "MatGetBlockSize returned bSize=%" PetscInt_FMT " (must be >= 1)", amgx->bSize);
    PetscCheck(amgx->nLocalRows % amgx->bSize == 0, amgx->comm, PETSC_ERR_ARG_WRONG,
               "Local row count %" PetscInt_FMT " is not divisible by advertised block size %" PetscInt_FMT
               "; PCAMGX cannot repack a partial block row.",
               amgx->nLocalRows, amgx->bSize);

    // XXX Currently constrained to 32-bit indices, to be changed in the future
    // Create the distribution and upload the matrix data
    AMGX_distribution_handle dist;
    PetscCallAmgX(AMGX_distribution_create(&dist, amgx->cfg));
    PetscCallAmgX(AMGX_distribution_set_32bit_colindices(dist, true));

    if (amgx->bSize == 1) {
      /* Scalar path. Pass scalar partition offsets and scalar CSR directly.
         (For the GPU case, rowOffsets/colIndices are host pointers but
         values is a device pointer; AMGX's upload internally detects each
         pointer's residency via cudaPointerGetAttributes.) */
      amgx->nnzBlocks = amgx->nnz;
      PetscCallAmgX(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, partitionOffsets.data()));
      PetscCallAmgX(AMGX_matrix_upload_distributed(amgx->A,
                                                   amgx->nGlobalRows,
                                                   (int)amgx->nLocalRows,
                                                   (int)amgx->nnz,
                                                   1, 1,
                                                   rowOffsets, colIndices, amgx->values,
                                                   NULL, dist));
    } else {
      /* Block path. AMGX wants block-coordinate row/col indices, block-major
         values, and block-coordinate partition offsets. The MATAIJ scalar CSR
         we have here uses scalar coordinates, so we repack on the host. */
      const PetscScalar       *hostValues = nullptr;
      std::vector<PetscScalar> hostValuesBuf;
      if (is_dev_ptrs) {
        hostValuesBuf.resize((size_t)amgx->nnz);
        PetscCallCUDA(cudaMemcpy(hostValuesBuf.data(), amgx->values,
                                 (size_t)amgx->nnz * sizeof(PetscScalar),
                                 cudaMemcpyDeviceToHost));
        hostValues = hostValuesBuf.data();
      } else {
        hostValues = amgx->values;
      }

      std::vector<int>         blockRowOff;
      std::vector<int>         blockColIdx;
      std::vector<PetscScalar> blockValues;
      PetscInt                 nnzBlocks = 0;
      PetscCall(PCAMGXBuildBlockCSR_Private(amgx->bSize, amgx->nLocalRows,
                                            rowOffsets, colIndices, hostValues,
                                            blockRowOff, blockColIdx, blockValues,
                                            &nnzBlocks));
      amgx->nnzBlocks = nnzBlocks;

      /* All ranks' partition boundaries must align to bSize because block rows
         are inseparable across rank boundaries. */
      std::vector<int> blockPartitionOffsets(amgx->nranks + 1);
      for (PetscInt r = 0; r <= amgx->nranks; ++r) {
        PetscCheck(partitionOffsets[r] % amgx->bSize == 0, amgx->comm, PETSC_ERR_PLIB,
                   "Partition offset %" PetscInt_FMT " (rank %" PetscInt_FMT
                   ") is not aligned to bSize=%" PetscInt_FMT
                   "; block rows must not straddle rank boundaries.",
                   partitionOffsets[r], r, amgx->bSize);
        blockPartitionOffsets[(size_t)r] = (int)(partitionOffsets[r] / amgx->bSize);
      }

      const int nBlockRowsGlobal = (int)(amgx->nGlobalRows / amgx->bSize);
      const int nBlockRowsLocal  = (int)(amgx->nLocalRows / amgx->bSize);

      PetscCallAmgX(AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, blockPartitionOffsets.data()));
      PetscCallAmgX(AMGX_matrix_upload_distributed(amgx->A,
                                                   nBlockRowsGlobal,
                                                   nBlockRowsLocal,
                                                   (int)nnzBlocks,
                                                   (int)amgx->bSize, (int)amgx->bSize,
                                                   blockRowOff.data(), blockColIdx.data(), blockValues.data(),
                                                   NULL, dist));
    }

    PetscCallAmgX(AMGX_solver_setup(amgx->solver, amgx->A));
    PetscCallAmgX(AMGX_vector_bind(amgx->sol, amgx->A));
    PetscCallAmgX(AMGX_vector_bind(amgx->rhs, amgx->A));

    PetscInt nlr = 0;
    PetscCall(MatRestoreRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &nlr, &rowOffsets, &colIndices, &done));
  } else {
    // The fast path for if the sparsity pattern persists
    if (amgx->bSize == 1) {
      PetscCallAmgX(AMGX_matrix_replace_coefficients(amgx->A, amgx->nLocalRows, amgx->nnz, amgx->values, NULL));
    } else {
      /* Sparsity persists, but values changed. Repack the new scalar values
         into block-major form. (We re-derive the block sparsity pattern; an
         optional optimization is to cache the rowOff/colIdx mapping from the
         slow path and only reshuffle values, but the cost of re-derivation is
         small relative to GPU upload + solve.) */
      PetscBool       done;
      const PetscInt *colIndices;
      const PetscInt *rowOffsets;
      PetscInt        nLocalRowsCheck = 0;
      PetscCall(MatGetRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &nLocalRowsCheck, &rowOffsets, &colIndices, &done));
      PetscCheck(done, amgx->comm, PETSC_ERR_PLIB, "MatGetRowIJ was not successful (fast path)");

      const PetscScalar       *hostValues = nullptr;
      std::vector<PetscScalar> hostValuesBuf;
      if (is_dev_ptrs) {
        hostValuesBuf.resize((size_t)amgx->nnz);
        PetscCallCUDA(cudaMemcpy(hostValuesBuf.data(), amgx->values,
                                 (size_t)amgx->nnz * sizeof(PetscScalar),
                                 cudaMemcpyDeviceToHost));
        hostValues = hostValuesBuf.data();
      } else {
        hostValues = amgx->values;
      }

      std::vector<int>         blockRowOff;
      std::vector<int>         blockColIdx;
      std::vector<PetscScalar> blockValues;
      PetscInt                 nnzBlocks = 0;
      PetscCall(PCAMGXBuildBlockCSR_Private(amgx->bSize, amgx->nLocalRows,
                                            rowOffsets, colIndices, hostValues,
                                            blockRowOff, blockColIdx, blockValues,
                                            &nnzBlocks));
      PetscCheck(nnzBlocks == amgx->nnzBlocks, amgx->comm, PETSC_ERR_PLIB,
                 "PCAMGX fast path: block-nnz changed (was %" PetscInt_FMT
                 ", now %" PetscInt_FMT
                 "); the caller marked SAME_NONZERO_PATTERN but the sparsity "
                 "differs from the previous setup.",
                 amgx->nnzBlocks, nnzBlocks);

      const int nBlockRowsLocal = (int)(amgx->nLocalRows / amgx->bSize);
      PetscCallAmgX(AMGX_matrix_replace_coefficients(amgx->A,
                                                     nBlockRowsLocal,
                                                     (int)nnzBlocks,
                                                     blockValues.data(), NULL));

      PetscInt nlr = 0;
      PetscCall(MatRestoreRowIJ(amgx->localA, 0, PETSC_FALSE, PETSC_FALSE, &nlr, &rowOffsets, &colIndices, &done));
    }
    PetscCallAmgX(AMGX_solver_resetup(amgx->solver, amgx->A));
  }

  if (is_dev_ptrs) {
    PetscCall(MatSeqAIJCUSPARSERestoreArrayRead(amgx->localA, &amgx->values));
  } else {
    PetscCall(MatSeqAIJRestoreArrayRead(amgx->localA, &amgx->values));
  }
  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCApply_AMGX - Applies the AmgX preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  b - rhs vector

   Output Parameter:
.  x - solution vector

   Application Interface Routine: PCApply()
 */
static PetscErrorCode PCApply_AMGX(PC pc, Vec b, Vec x)
{
  PC_AMGX           *amgx = (PC_AMGX *)pc->data;
  PetscScalar       *x_;
  const PetscScalar *b_;
  PetscBool          is_dev_ptrs;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &is_dev_ptrs, VECCUDA, VECMPICUDA, VECSEQCUDA, ""));

  if (is_dev_ptrs) {
    PetscCall(VecCUDAGetArrayWrite(x, &x_));
    PetscCall(VecCUDAGetArrayRead(b, &b_));
  } else {
    PetscCall(VecGetArrayWrite(x, &x_));
    PetscCall(VecGetArrayRead(b, &b_));
  }

  /* AMGX_vector_upload(handle, n, block_dim, data) takes n = number of block
     entries and block_dim = block size, with total scalars = n * block_dim.
     For bSize == 1 this reduces to the scalar form (n = nLocalRows,
     block_dim = 1). The vector dims must agree with the AMGX matrix's block
     layout, which is set in PCSetUp_AMGX. */
  const int nBlockRowsLocal = (int)(amgx->nLocalRows / amgx->bSize);
  PetscCallAmgX(AMGX_vector_upload(amgx->sol, nBlockRowsLocal, (int)amgx->bSize, x_));
  PetscCallAmgX(AMGX_vector_upload(amgx->rhs, nBlockRowsLocal, (int)amgx->bSize, b_));
  PetscCallAmgX(AMGX_solver_solve_with_0_initial_guess(amgx->solver, amgx->rhs, amgx->sol));

  AMGX_SOLVE_STATUS status;
  PetscCallAmgX(AMGX_solver_get_status(amgx->solver, &status));
  PetscCall(PCSetErrorIfFailure(pc, static_cast<PetscBool>(status == AMGX_SOLVE_FAILED)));
  PetscCheck(status != AMGX_SOLVE_FAILED, amgx->comm, PETSC_ERR_CONV_FAILED, "AmgX solver failed to solve the system! The error code is %d.", status);
  PetscCallAmgX(AMGX_vector_download(amgx->sol, x_));

  if (is_dev_ptrs) {
    PetscCall(VecCUDARestoreArrayWrite(x, &x_));
    PetscCall(VecCUDARestoreArrayRead(b, &b_));
  } else {
    PetscCall(VecRestoreArrayWrite(x, &x_));
    PetscCall(VecRestoreArrayRead(b, &b_));
  }
  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (amgx->solve_state_init) {
    PetscCallAmgX(AMGX_solver_destroy(amgx->solver));
    PetscCallAmgX(AMGX_matrix_destroy(amgx->A));
    PetscCallAmgX(AMGX_vector_destroy(amgx->sol));
    PetscCallAmgX(AMGX_vector_destroy(amgx->rhs));
    if (amgx->nranks > 1) PetscCall(MatDestroy(&amgx->localA));
    PetscCall(amgx_output_messages(amgx));
    amgx->solve_state_init = false;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PCDestroy_AMGX - Destroys the private context for the AmgX preconditioner
   that was created with PCCreate_AMGX().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  /* Each teardown is guarded on whether the corresponding handle was
     actually allocated. The setup path can fail at multiple intermediate
     points (e.g. invalid option in PCSetFromOptions_AMGX, or
     AMGX_resources_create not yet reached in PCSetUp_AMGX), leaving the
     PC partially initialized. The destructor must tolerate that without
     itself raising a second error and triggering MPI_Abort.

     The struct was allocated via PetscNew (zero-fills) in PCCreate_AMGX,
     so all handle members default to null / MPI_COMM_NULL until set. */
  /* decrease the number of instances, only the last instance need to destroy resource and finalizing AmgX */
  if (s_count == 1) {
    /* can put this in a PCAMGXInitializePackage method */
    if (amgx->rsrc) {
      PetscCallAmgX(AMGX_resources_destroy(amgx->rsrc));
      amgx->rsrc = nullptr;
    }
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    if (amgx->cfg) {
      PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
      amgx->cfg = nullptr;
    }
    PetscCallAmgX(AMGX_finalize_plugins());
    PetscCallAmgX(AMGX_finalize());
    if (amgx->comm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&amgx->comm));
  } else {
    if (amgx->cfg) {
      PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
      amgx->cfg = nullptr;
    }
  }
  s_count -= 1;
  PetscCall(PetscFree(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class T>
std::string map_reverse_lookup(const std::map<std::string, T> &map, const T &key)
{
  for (auto const &m : map) {
    if (m.second == key) return m.first;
  }
  return "";
}

static PetscErrorCode PCSetFromOptions_AMGX(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_AMGX      *amgx          = (PC_AMGX *)pc->data;
  constexpr int MAX_PARAM_LEN = 128;
  char          option[MAX_PARAM_LEN];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "AmgX options");
  amgx->cfg_contents = "config_version=2,";
  amgx->cfg_contents += "determinism_flag=1,";

  // Set exact coarse solve
  PetscCall(PetscOptionsBool("-pc_amgx_exact_coarse_solve", "AmgX AMG Exact Coarse Solve", "", amgx->exact_coarse_solve, &amgx->exact_coarse_solve, NULL));
  if (amgx->exact_coarse_solve) amgx->cfg_contents += "exact_coarse_solve=1,";

  amgx->cfg_contents += "solver(amg)=AMG,";

  // Set method
  std::string def_amg_method = map_reverse_lookup(AmgXControlMap::AMGMethods, amgx->amg_method);
  PetscCall(PetscStrncpy(option, def_amg_method.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_amg_method", "AmgX AMG Method", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGMethods.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Method %s not registered for AmgX.", option);
  amgx->amg_method = AmgXControlMap::AMGMethods.at(option);
  amgx->cfg_contents += "amg:algorithm=" + std::string(option) + ",";

  // Set cycle
  std::string def_amg_cycle = map_reverse_lookup(AmgXControlMap::AMGCycles, amgx->amg_cycle);
  PetscCall(PetscStrncpy(option, def_amg_cycle.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_amg_cycle", "AmgX AMG Cycle", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::AMGCycles.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "AMG Cycle %s not registered for AmgX.", option);
  amgx->amg_cycle = AmgXControlMap::AMGCycles.at(option);
  amgx->cfg_contents += "amg:cycle=" + std::string(option) + ",";

  // Set smoother
  std::string def_smoother = map_reverse_lookup(AmgXControlMap::Smoothers, amgx->smoother);
  PetscCall(PetscStrncpy(option, def_smoother.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_smoother", "AmgX Smoother", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::Smoothers.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Smoother %s not registered for AmgX.", option);
  amgx->smoother = AmgXControlMap::Smoothers.at(option);
  amgx->cfg_contents += "amg:smoother(smooth)=" + std::string(option) + ",";

  if (amgx->smoother == AmgXSmoother::JacobiL1 || amgx->smoother == AmgXSmoother::BlockJacobi) {
    PetscCall(PetscOptionsScalar("-pc_amgx_jacobi_relaxation_factor", "AmgX AMG Jacobi Relaxation Factor", "", amgx->jacobi_relaxation_factor, &amgx->jacobi_relaxation_factor, NULL));
    amgx->cfg_contents += "smooth:relaxation_factor=" + std::to_string(amgx->jacobi_relaxation_factor) + ",";
  } else if (amgx->smoother == AmgXSmoother::GS || amgx->smoother == AmgXSmoother::MulticolorGS) {
    PetscCall(PetscOptionsScalar("-pc_amgx_gs_symmetric", "AmgX AMG Gauss Seidel Symmetric", "", amgx->gs_symmetric, &amgx->gs_symmetric, NULL));
    amgx->cfg_contents += "smooth:symmetric_GS=" + std::to_string(amgx->gs_symmetric) + ",";
  }

  // Set selector
  std::string def_selector = map_reverse_lookup(AmgXControlMap::Selectors, amgx->selector);
  PetscCall(PetscStrncpy(option, def_selector.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_selector", "AmgX Selector", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::Selectors.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Selector %s not registered for AmgX.", option);

  /* Validate the selector against the AMG method. Use the just-parsed
     selector (held in `option`), not the cached `amgx->selector` from a
     previous PCSetFromOptions call — otherwise switching from CLASSICAL
     (default selector PMIS) to AGGREGATION + an explicit selector like
     SIZE_2 is rejected even though both are valid in isolation. */
  const AmgXSelector parsed_selector = AmgXControlMap::Selectors.at(option);
  if (amgx->amg_method == AmgXAMGMethod::Classical) {
    PetscCheck(parsed_selector == AmgXSelector::PMIS || parsed_selector == AmgXSelector::HMIS, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Classical AMG: selector=%s", option);
    amgx->cfg_contents += "amg:interpolator=D2,";
  } else if (amgx->amg_method == AmgXAMGMethod::Aggregation) {
    PetscCheck(parsed_selector == AmgXSelector::Size2 || parsed_selector == AmgXSelector::Size4 || parsed_selector == AmgXSelector::Size8 || parsed_selector == AmgXSelector::MultiPairwise, amgx->comm, PETSC_ERR_PLIB, "Chosen selector is not used for AmgX Aggregation AMG: selector=%s (valid: SIZE_2, SIZE_4, SIZE_8, MULTI_PAIRWISE)", option);
  }
  amgx->selector = parsed_selector;
  amgx->cfg_contents += "amg:selector=" + std::string(option) + ",";

  // Set presweeps
  PetscCall(PetscOptionsInt("-pc_amgx_presweeps", "AmgX AMG Presweep Count", "", amgx->presweeps, &amgx->presweeps, NULL));
  amgx->cfg_contents += "amg:presweeps=" + std::to_string(amgx->presweeps) + ",";

  // Set postsweeps
  PetscCall(PetscOptionsInt("-pc_amgx_postsweeps", "AmgX AMG Postsweep Count", "", amgx->postsweeps, &amgx->postsweeps, NULL));
  amgx->cfg_contents += "amg:postsweeps=" + std::to_string(amgx->postsweeps) + ",";

  // Set max levels
  PetscCall(PetscOptionsInt("-pc_amgx_max_levels", "AmgX AMG Max Level Count", "", amgx->max_levels, &amgx->max_levels, NULL));
  amgx->cfg_contents += "amg:max_levels=" + std::to_string(amgx->max_levels) + ",";

  // Set dense LU num rows
  PetscCall(PetscOptionsInt("-pc_amgx_dense_lu_num_rows", "AmgX Dense LU Number of Rows", "", amgx->dense_lu_num_rows, &amgx->dense_lu_num_rows, NULL));
  amgx->cfg_contents += "amg:dense_lu_num_rows=" + std::to_string(amgx->dense_lu_num_rows) + ",";

  // Set strength threshold
  PetscCall(PetscOptionsScalar("-pc_amgx_strength_threshold", "AmgX AMG Strength Threshold", "", amgx->strength_threshold, &amgx->strength_threshold, NULL));
  amgx->cfg_contents += "amg:strength_threshold=" + std::to_string(amgx->strength_threshold) + ",";

  // Set aggressive_levels
  PetscCall(PetscOptionsInt("-pc_amgx_aggressive_levels", "AmgX AMG Presweep Count", "", amgx->aggressive_levels, &amgx->aggressive_levels, NULL));
  if (amgx->aggressive_levels > 0) amgx->cfg_contents += "amg:aggressive_levels=" + std::to_string(amgx->aggressive_levels) + ",";

  // Set coarse solver
  std::string def_coarse_solver = map_reverse_lookup(AmgXControlMap::CoarseSolvers, amgx->coarse_solver);
  PetscCall(PetscStrncpy(option, def_coarse_solver.c_str(), sizeof(option)));
  PetscCall(PetscOptionsString("-pc_amgx_coarse_solver", "AmgX CoarseSolver", "", option, option, MAX_PARAM_LEN, NULL));
  PetscCheck(AmgXControlMap::CoarseSolvers.count(option) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "CoarseSolver %s not registered for AmgX.", option);
  amgx->coarse_solver = AmgXControlMap::CoarseSolvers.at(option);
  amgx->cfg_contents += "amg:coarse_solver=" + std::string(option) + ",";

  // Set max iterations
  amgx->cfg_contents += "amg:max_iters=1,";

  // Set output control parameters
  PetscCall(PetscOptionsBool("-pc_amgx_print_grid_stats", "AmgX Print Grid Stats", "", amgx->print_grid_stats, &amgx->print_grid_stats, NULL));

  if (amgx->print_grid_stats) amgx->cfg_contents += "amg:print_grid_stats=1,";
  amgx->cfg_contents += "amg:monitor_residual=0";

  // Set whether AmgX output will be seen
  PetscCall(PetscOptionsBool("-pc_amgx_verbose", "Enable output from AmgX", "", amgx->verbose, &amgx->verbose, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_AMGX(PC pc, PetscViewer viewer)
{
  PC_AMGX  *amgx = (PC_AMGX *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    std::string output_cfg(amgx->cfg_contents);
    std::replace(output_cfg.begin(), output_cfg.end(), ',', '\n');
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n%s\n", output_cfg.c_str()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PCAMGX - Interface to NVIDIA's AmgX algebraic multigrid

  Options Database Keys:
+    -pc_amgx_amg_method (CLASSICAL,AGGREGATION)                       - set the AMG algorithm to use
.    -pc_amgx_amg_cycle (V,W,F,CG)                                     - set the AMG cycle type
.    -pc_amgx_jacobi_relaxation_factor                                 - set the relaxation factor for Jacobi smoothing
.    -pc_amgx_gs_symmetric                                             - enforce symmetric Gauss-Seidel smoothing (only applies if GS smoothing is selected)
. -pc_amgx_selector (SIZE_2|SIZE_4|SIZE_8|MULTI_PAIRWISE|PMIS|HMIS) - set the AMG coarse selector
.    -pc_amgx_presweeps                                                - set the number of AMG pre-sweeps
.    -pc_amgx_postsweeps                                               - set the number of AMG post-sweeps
.    -pc_amgx_max_levels                                               - set the maximum number of levels in the AMG level hierarchy
.    -pc_amgx_strength_threshold                                       - set the strength threshold for the AMG coarsening
.    -pc_amgx_aggressive_levels                                        - set the number of levels (from the finest) that should apply aggressive coarsening
.    -pc_amgx_coarse_solver (DENSE_LU_SOLVER,NOSOLVER)                 - set the coarse solve
.    -pc_amgx_print_grid_stats                                         - output the AMG grid hierarchy to `stdout`
-    -pc_amgx_verbose                                                  - enable AmgX verbose output
-    -pc_amgx_smoother (PCG|PCGF|PBICGSTAB|GMRES|FGMRES|JACOBI_L1|BLOCK_JACOBI|GS|MULTICOLOR_GS|MULTICOLOR_ILU|MULTICOLOR_DILU|CHEBYSHEV_POLY|NOSOLVER) - set the AMG pre/post smoother

   Level: intermediate

   Note:
   Implementation will accept host or device pointers, but good performance will require that the `KSP` is also GPU accelerated so that data is not frequently transferred between host and device.

.seealso: [](ch_ksp), `PCGAMG`, `PCHYPRE`, `PCMG`, `PCAmgXGetResources()`, `PCCreate()`, `PCSetType()`, `PCType`, `PC`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_AMGX(PC pc)
{
  PC_AMGX *amgx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&amgx));
  pc->ops->apply          = PCApply_AMGX;
  pc->ops->setfromoptions = PCSetFromOptions_AMGX;
  pc->ops->setup          = PCSetUp_AMGX;
  pc->ops->view           = PCView_AMGX;
  pc->ops->destroy        = PCDestroy_AMGX;
  pc->ops->reset          = PCReset_AMGX;
  pc->data                = (void *)amgx;

  // Set the defaults
  amgx->selector                 = AmgXSelector::PMIS;
  amgx->smoother                 = AmgXSmoother::BlockJacobi;
  amgx->amg_method               = AmgXAMGMethod::Classical;
  amgx->coarse_solver            = AmgXCoarseSolver::DenseLU;
  amgx->amg_cycle                = AmgXAMGCycle::V;
  amgx->exact_coarse_solve       = PETSC_TRUE;
  amgx->presweeps                = 1;
  amgx->postsweeps               = 1;
  amgx->max_levels               = 100;
  amgx->strength_threshold       = 0.5;
  amgx->aggressive_levels        = 0;
  amgx->dense_lu_num_rows        = 1;
  amgx->jacobi_relaxation_factor = 0.9;
  amgx->gs_symmetric             = PETSC_FALSE;
  amgx->print_grid_stats         = PETSC_FALSE;
  amgx->verbose                  = PETSC_FALSE;
  amgx->rsrc_init                = false;
  amgx->solve_state_init         = false;

  s_count++;

  PetscCallCUDA(cudaGetDevice(&amgx->devID));
  if (s_count == 1) {
    PetscCallAmgX(AMGX_initialize());
    PetscCallAmgX(AMGX_initialize_plugins());
    PetscCallAmgX(AMGX_register_print_callback(&print_callback));
    PetscCallAmgX(AMGX_install_signal_handler());
  }
  /* This communicator is not yet known to this system, so we duplicate it and make an internal communicator */
  PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)pc), &amgx->comm));
  PetscCallMPI(MPI_Comm_size(amgx->comm, &amgx->nranks));
  PetscCallMPI(MPI_Comm_rank(amgx->comm, &amgx->rank));

  PetscCall(amgx_output_messages(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCAmgXGetResources - get AMGx's internal resource object

  Not Collective, No Fortran Support

  Input Parameter:
. pc - the PC

  Output Parameter:
. rsrc_out - pointer to the AMGx resource object

  Level: advanced

.seealso: [](ch_ksp), `PCAMGX`, `PC`, `PCGAMG`
@*/
PETSC_EXTERN PetscErrorCode PCAmgXGetResources(PC pc, void *rsrc_out)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (!amgx->rsrc_init) {
    // Read configuration file
    PetscCallAmgX(AMGX_config_create(&amgx->cfg, amgx->cfg_contents.c_str()));
    PetscCallAmgX(AMGX_resources_create(&amgx->rsrc, amgx->cfg, &amgx->comm, 1, &amgx->devID));
    amgx->rsrc_init = true;
  }
  *static_cast<AMGX_resources_handle *>(rsrc_out) = amgx->rsrc;
  PetscFunctionReturn(PETSC_SUCCESS);
}
