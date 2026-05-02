/*
   Small-dense host-side linear-algebra helpers for KSPGMSTAB.

   All matrices are column-major PetscScalar* arrays. Sizes are typically
   bounded by 2s+3 ≤ 11 for default s=4, so direct LAPACK calls beat any
   fancier blocking. Exact correspondence with the C++ port helpers is
   documented per function.
*/
#pragma once
#include <petscsys.h>
#include <petscblaslapack.h>

/* ---- Small-dense ops ----------------------------------------------------- */

/* MATLAB:  [Q, L] = lq(Z)  s.t. Z = L * Q'.
   Z is m x n, column-major, ldZ ≥ m.  Q is n x n orthogonal,
   L is m x n lower-triangular. Outputs are caller-allocated:
     Q[n*n], L[m*n].

   Mirrors gmstab_cpp/src/small_dense.cpp::lq exactly:
       Zt = Z'                 (n x m)
       [Qzt, Rzt] = qr(Zt)     Eigen HouseholderQR; LAPACK dgeqrf+dorgqr
       Q = Qzt                 (n x n full Q)
       R = Rzt(0:n, 0:m)       (n x m upper-tri block)
       L = R'                  (m x n lower-tri)
*/
PETSC_INTERN PetscErrorCode KSPGMSTABLq_Private(const PetscScalar *Z, PetscInt ldZ,
                                                 PetscInt m, PetscInt n,
                                                 PetscScalar *Q, PetscInt ldQ,
                                                 PetscScalar *L, PetscInt ldL);

/* MATLAB:  out = orth(M)
   M is m x n, returns m x rank columns of an orthonormal basis for the
   range of M. Uses SVD-based rank determination (matches Eigen's
   BDCSVD path used by gmstab_cpp::orth).

   Tolerance: max(m, n) * eps * sigma_max.

   The output Q has out_cols ≤ n columns; caller passes a workspace at
   least (m × n) and the function writes (m × out_cols). out_cols is
   reported via *rank_out.
*/
PETSC_INTERN PetscErrorCode KSPGMSTABOrth_Private(const PetscScalar *M, PetscInt ldM,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar *Q, PetscInt ldQ,
                                                   PetscInt *rank_out);

/* MATLAB:  out = nullbasis(H)  := full Q from qr(H), columns n+1..m
   H is m x n (m > n). Output is m x (m-n).
*/
PETSC_INTERN PetscErrorCode KSPGMSTABNullBasis_Private(const PetscScalar *H, PetscInt ldH,
                                                        PetscInt m, PetscInt n,
                                                        PetscScalar *Qtail, PetscInt ldQtail);

/* C := alpha * A * B + beta * C — column-major dgemm wrapper. */
PETSC_INTERN PetscErrorCode KSPGMSTABGemm_Private(const char *transA, const char *transB,
                                                   PetscInt m, PetscInt n, PetscInt k,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   const PetscScalar *B, PetscInt ldB,
                                                   PetscScalar beta,
                                                   PetscScalar *C, PetscInt ldC);

/* y := alpha * A * x + beta * y — column-major dgemv wrapper. */
PETSC_INTERN PetscErrorCode KSPGMSTABGemv_Private(const char *transA,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   const PetscScalar *x, PetscInt incx,
                                                   PetscScalar beta,
                                                   PetscScalar *y, PetscInt incy);

/* trsm: in-place triangular solve. side='L' or 'R', uplo='L' or 'U'. */
PETSC_INTERN PetscErrorCode KSPGMSTABTrsm_Private(const char *side, const char *uplo,
                                                   const char *trans, const char *diag,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   PetscScalar *B, PetscInt ldB);

/* trsv: triangular x = T \ x in place. uplo='L' / 'U'. */
PETSC_INTERN PetscErrorCode KSPGMSTABTrsv_Private(const char *uplo, const char *trans, const char *diag,
                                                   PetscInt n,
                                                   const PetscScalar *T, PetscInt ldT,
                                                   PetscScalar *x, PetscInt incx);

/* General LU solve A*X = B. A is n x n, B is n x nrhs, both column-major.
   X is returned in B; A is destroyed (LU factors written into it). */
PETSC_INTERN PetscErrorCode KSPGMSTABGesv_Private(PetscInt n, PetscInt nrhs,
                                                   PetscScalar *A, PetscInt ldA,
                                                   PetscScalar *B, PetscInt ldB);

/* QR-based least squares solve A * x = b for tall A (m >= n). Equivalent
   to Eigen's householderQr().solve(b). m, n are A's sizes; A is m x n,
   b is length m, x is length n. A and b are destroyed in place; x is
   read out of b's first n entries on return. */
PETSC_INTERN PetscErrorCode KSPGMSTABQrLeastSquares_Private(PetscInt m, PetscInt n,
                                                             PetscScalar *A, PetscInt ldA,
                                                             PetscScalar *b);

/* Compute the Frobenius / 2-norm of a small dense column vector. Trivial,
   but having a wrapper keeps the cycle code readable. */
PETSC_INTERN PetscReal KSPGMSTABNrm2_Private(PetscInt n, const PetscScalar *x, PetscInt incx);
