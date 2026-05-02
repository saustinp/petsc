/*
   Small-dense host-side linear-algebra helpers for KSPGMSTAB. See the
   header for contracts. All implementations are LAPACK / BLAS thin
   wrappers; correctness depends entirely on matching the reduction
   conventions Eigen uses (which are themselves LAPACK-equivalent).
*/
#include <petsc/private/petscimpl.h>
#include <petscblaslapack.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_smalldense.h>
#include <math.h>
#include <string.h>

/* ============================================================================
 * BLAS-level wrappers
 * ============================================================================ */

PETSC_INTERN PetscErrorCode KSPGMSTABGemm_Private(const char *transA, const char *transB,
                                                   PetscInt m, PetscInt n, PetscInt k,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   const PetscScalar *B, PetscInt ldB,
                                                   PetscScalar beta,
                                                   PetscScalar *C, PetscInt ldC)
{
  PetscBLASInt m_, n_, k_, lda_, ldb_, ldc_;
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(k, &k_));
  PetscCall(PetscBLASIntCast(ldA, &lda_));
  PetscCall(PetscBLASIntCast(ldB, &ldb_));
  PetscCall(PetscBLASIntCast(ldC, &ldc_));
  PetscCallBLAS("BLASgemm", BLASgemm_(transA, transB, &m_, &n_, &k_, &alpha, (PetscScalar *)A, &lda_, (PetscScalar *)B, &ldb_, &beta, C, &ldc_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABGemv_Private(const char *transA,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   const PetscScalar *x, PetscInt incx,
                                                   PetscScalar beta,
                                                   PetscScalar *y, PetscInt incy)
{
  PetscBLASInt m_, n_, lda_, incx_, incy_;
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(ldA, &lda_));
  PetscCall(PetscBLASIntCast(incx, &incx_));
  PetscCall(PetscBLASIntCast(incy, &incy_));
  PetscCallBLAS("BLASgemv", BLASgemv_(transA, &m_, &n_, &alpha, (PetscScalar *)A, &lda_, (PetscScalar *)x, &incx_, &beta, y, &incy_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABTrsm_Private(const char *side, const char *uplo,
                                                   const char *trans, const char *diag,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar alpha,
                                                   const PetscScalar *A, PetscInt ldA,
                                                   PetscScalar *B, PetscInt ldB)
{
  PetscBLASInt m_, n_, lda_, ldb_;
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(ldA, &lda_));
  PetscCall(PetscBLASIntCast(ldB, &ldb_));
  PetscCallBLAS("BLAStrsm", BLAStrsm_(side, uplo, trans, diag, &m_, &n_, &alpha, (PetscScalar *)A, &lda_, B, &ldb_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABTrsv_Private(const char *uplo, const char *trans, const char *diag,
                                                   PetscInt n,
                                                   const PetscScalar *T, PetscInt ldT,
                                                   PetscScalar *x, PetscInt incx)
{
  PetscBLASInt n_, ldT_, incx_;
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(ldT, &ldT_));
  PetscCall(PetscBLASIntCast(incx, &incx_));
  PetscCallBLAS("BLAStrsv", BLAStrsv_(uplo, trans, diag, &n_, (PetscScalar *)T, &ldT_, x, &incx_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABGesv_Private(PetscInt n, PetscInt nrhs,
                                                   PetscScalar *A, PetscInt ldA,
                                                   PetscScalar *B, PetscInt ldB)
{
  PetscBLASInt n_, nrhs_, lda_, ldb_, info;
  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(nrhs, &nrhs_));
  PetscCall(PetscBLASIntCast(ldA, &lda_));
  PetscCall(PetscBLASIntCast(ldB, &ldb_));
  PetscBLASInt *ipiv;
  PetscCall(PetscMalloc1(n, &ipiv));
  PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&n_, &nrhs_, A, &lda_, ipiv, B, &ldb_, &info));
  PetscCall(PetscFree(ipiv));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgesv: info = %d", (int)info);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscReal KSPGMSTABNrm2_Private(PetscInt n, const PetscScalar *x, PetscInt incx)
{
  PetscReal s = 0.0;
  for (PetscInt i = 0; i < n; ++i) {
    PetscReal v = PetscRealPart(x[(size_t)i * (size_t)incx]);
    s += v * v;
  }
  return PetscSqrtReal(s);
}

/* ============================================================================
 * KSPGMSTABLq_Private
 *   Z (m x n)  →  L (m x n, lower-tri),  Q (n x n, orthogonal)  s.t. Z = L*Q'.
 *
 * Strategy mirrors Eigen / C++:
 *   1) Form Zt = Z' (n x m).
 *   2) DGEQRF on Zt: Householder reflectors land in Zt; tau holds scalars.
 *   3) Extract upper triangle of Zt(0:n, 0:m): that's R (n x m).
 *   4) Take L = R' (m x n, lower-tri).
 *   5) DORGQR to produce the explicit orthogonal Q (n x n) into a buffer.
 *
 * LAPACK convention:
 *   - For an n x m matrix with n ≤ m, DGEQRF produces R as the (min(n,m)
 *     = n) x m upper-triangular block (rectangular form). DORGQR needs
 *     enough columns to span the full Q; for n x n Q we ask for n cols.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABLq_Private(const PetscScalar *Z, PetscInt ldZ,
                                                 PetscInt m, PetscInt n,
                                                 PetscScalar *Q, PetscInt ldQ,
                                                 PetscScalar *L, PetscInt ldL)
{
  PetscFunctionBegin;
  PetscCheck(m > 0 && n > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "lq: m=%" PetscInt_FMT " n=%" PetscInt_FMT, m, n);

  /* (1) Allocate Zt and copy Z' into it. Zt is n x m, ldim = n. */
  PetscScalar *Zt;
  const PetscInt ldZt = n;
  PetscCall(PetscMalloc1((size_t)n * (size_t)m, &Zt));
  for (PetscInt i = 0; i < m; ++i)
    for (PetscInt j = 0; j < n; ++j)
      Zt[(size_t)j + (size_t)i * (size_t)ldZt] = Z[(size_t)i + (size_t)j * (size_t)ldZ];

  /* (2) DGEQRF on Zt. Workspace query then alloc. */
  PetscBLASInt n_, m_, ldZt_, lwork, info;
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(ldZt, &ldZt_));

  PetscScalar *tau;
  PetscCall(PetscMalloc1(PetscMin(n, m), &tau));

  PetscScalar wk_query;
  lwork = -1;
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&n_, &m_, Zt, &ldZt_, tau, &wk_query, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgeqrf workspace query: info=%d", (int)info);
  lwork = (PetscBLASInt)PetscRealPart(wk_query);
  PetscScalar *work;
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&n_, &m_, Zt, &ldZt_, tau, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgeqrf: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* (3,4) Extract R (upper triangle of Zt(0:n, 0:m)) and write L = R'. */
  for (PetscInt j = 0; j < n; ++j)
    for (PetscInt i = 0; i < m; ++i)
      L[(size_t)i + (size_t)j * (size_t)ldL] = 0.0;
  for (PetscInt i = 0; i < n; ++i)
    for (PetscInt j = i; j < m; ++j) {
      /* R(i, j) = Zt(i, j); L(j, i) = R(i, j). */
      const PetscScalar Rij = Zt[(size_t)i + (size_t)j * (size_t)ldZt];
      L[(size_t)j + (size_t)i * (size_t)ldL] = Rij;
    }

  /* (5) DORGQR to reify Q. */
  PetscBLASInt k_;
  PetscCall(PetscBLASIntCast(PetscMin(n, m), &k_));
  lwork = -1;
  PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&n_, &n_, &k_, Zt, &ldZt_, tau, &wk_query, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKorgqr query: info=%d", (int)info);
  lwork = (PetscBLASInt)PetscRealPart(wk_query);
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&n_, &n_, &k_, Zt, &ldZt_, tau, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKorgqr: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* Copy Zt (which now holds Q) into the caller's Q buffer. */
  for (PetscInt j = 0; j < n; ++j)
    for (PetscInt i = 0; i < n; ++i)
      Q[(size_t)i + (size_t)j * (size_t)ldQ] = Zt[(size_t)i + (size_t)j * (size_t)ldZt];

  PetscCall(PetscFree(tau));
  PetscCall(PetscFree(Zt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPGMSTABNullBasis_Private — full Q from QR, take trailing m-n columns.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABNullBasis_Private(const PetscScalar *H, PetscInt ldH,
                                                        PetscInt m, PetscInt n,
                                                        PetscScalar *Qtail, PetscInt ldQtail)
{
  PetscFunctionBegin;
  if (m <= n) PetscFunctionReturn(PETSC_SUCCESS);

  /* Copy H into Hbuf (m x n) for in-place factorization. */
  PetscScalar *Hbuf;
  PetscCall(PetscMalloc1((size_t)m * (size_t)m, &Hbuf));
  for (PetscInt j = 0; j < n; ++j)
    for (PetscInt i = 0; i < m; ++i)
      Hbuf[(size_t)i + (size_t)j * (size_t)m] = H[(size_t)i + (size_t)j * (size_t)ldH];
  /* Zero the trailing m-n columns so DORGQR has columns to expand into. */
  for (PetscInt j = n; j < m; ++j)
    for (PetscInt i = 0; i < m; ++i)
      Hbuf[(size_t)i + (size_t)j * (size_t)m] = 0.0;

  PetscBLASInt m_, n_, ldH_ = (PetscBLASInt)m, lwork, info, k_;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(n, &k_));

  PetscScalar *tau;
  PetscCall(PetscMalloc1(PetscMin(m, n), &tau));
  PetscScalar wk_q;
  lwork = -1;
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Hbuf, &ldH_, tau, &wk_q, &lwork, &info));
  lwork = (PetscBLASInt)PetscRealPart(wk_q);
  PetscScalar *work;
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Hbuf, &ldH_, tau, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgeqrf nullbasis: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* DORGQR with K = n (Householder reflectors), N = m (full Q is m x m). */
  lwork = -1;
  PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &m_, &k_, Hbuf, &ldH_, tau, &wk_q, &lwork, &info));
  lwork = (PetscBLASInt)PetscRealPart(wk_q);
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &m_, &k_, Hbuf, &ldH_, tau, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKorgqr nullbasis: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* Copy out columns n..m-1 (trailing m-n cols of Hbuf). */
  for (PetscInt j = n; j < m; ++j)
    for (PetscInt i = 0; i < m; ++i)
      Qtail[(size_t)i + (size_t)(j - n) * (size_t)ldQtail] = Hbuf[(size_t)i + (size_t)j * (size_t)m];

  PetscCall(PetscFree(tau));
  PetscCall(PetscFree(Hbuf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPGMSTABOrth_Private — SVD-based orthonormal basis (matches Eigen BDCSVD).
 *
 *   tol = max(m, n) * eps * sigma_max(M)
 *   rank = #{ s_i : s_i > tol }
 *   out = U(:, 0:rank)
 *
 * LAPACK: dgesdd (faster than dgesvd for general dense). Caller's Q
 * buffer must hold up to m x n.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABOrth_Private(const PetscScalar *M, PetscInt ldM,
                                                   PetscInt m, PetscInt n,
                                                   PetscScalar *Q, PetscInt ldQ,
                                                   PetscInt *rank_out)
{
  PetscFunctionBegin;
  if (m == 0 || n == 0) { *rank_out = 0; PetscFunctionReturn(PETSC_SUCCESS); }

  PetscScalar *A_;
  PetscCall(PetscMalloc1((size_t)m * (size_t)n, &A_));
  for (PetscInt j = 0; j < n; ++j)
    for (PetscInt i = 0; i < m; ++i)
      A_[(size_t)i + (size_t)j * (size_t)m] = M[(size_t)i + (size_t)j * (size_t)ldM];

  const PetscInt mn = PetscMin(m, n);
  PetscReal *S;
  PetscCall(PetscMalloc1(mn, &S));

  PetscScalar *U_;
  PetscCall(PetscMalloc1((size_t)m * (size_t)mn, &U_));     /* m x mn (thin U) */
  PetscScalar *Vt_dummy = NULL;                              /* not used; use jobu='S', jobvt='N' */

  PetscBLASInt m_, n_, ldA_, ldU_, ldV_ = 1, lwork, info;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(m, &ldA_));
  PetscCall(PetscBLASIntCast(m, &ldU_));

  PetscScalar wk_q;
  lwork = -1;
  /* dgesvd with jobu='S' produces U (m x min(m,n)), jobvt='N' skips Vt. */
  PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_("S", "N", &m_, &n_, A_, &ldA_, S, U_, &ldU_, Vt_dummy, &ldV_, &wk_q, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgesvd query: info=%d", (int)info);
  lwork = (PetscBLASInt)PetscRealPart(wk_q);
  PetscScalar *work;
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKgesvd", LAPACKgesvd_("S", "N", &m_, &n_, A_, &ldA_, S, U_, &ldU_, Vt_dummy, &ldV_, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgesvd: info=%d", (int)info);
  PetscCall(PetscFree(work));

  const PetscReal tol = ((PetscReal)PetscMax(m, n)) * PETSC_MACHINE_EPSILON * (mn > 0 ? S[0] : 0.0);
  PetscInt rank = 0;
  for (PetscInt i = 0; i < mn; ++i) if (S[i] > tol) ++rank;

  for (PetscInt j = 0; j < rank; ++j)
    for (PetscInt i = 0; i < m; ++i)
      Q[(size_t)i + (size_t)j * (size_t)ldQ] = U_[(size_t)i + (size_t)j * (size_t)m];
  *rank_out = rank;

  PetscCall(PetscFree(S));
  PetscCall(PetscFree(U_));
  PetscCall(PetscFree(A_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPGMSTABQrLeastSquares_Private — solve A * x = b for tall A.
 *
 *   1. DGEQRF on A → Householder.
 *   2. DORMQR: b := Q' * b.
 *   3. DTRSV with uplo='U' on the upper-triangular R block to solve R x = b(0:n).
 *   4. b's first n entries hold x.
 *
 * Mirrors Eigen's `householderQr().solve(b)`.
 * ============================================================================ */
PETSC_INTERN PetscErrorCode KSPGMSTABQrLeastSquares_Private(PetscInt m, PetscInt n,
                                                             PetscScalar *A, PetscInt ldA,
                                                             PetscScalar *b)
{
  PetscFunctionBegin;
  PetscCheck(m >= n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "QR LSQ: m=%" PetscInt_FMT " < n=%" PetscInt_FMT, m, n);

  PetscBLASInt m_, n_, lda_, lwork, info, one = 1;
  PetscCall(PetscBLASIntCast(m, &m_));
  PetscCall(PetscBLASIntCast(n, &n_));
  PetscCall(PetscBLASIntCast(ldA, &lda_));

  PetscScalar *tau;
  PetscCall(PetscMalloc1(n, &tau));
  PetscScalar wk_q;
  lwork = -1;
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, A, &lda_, tau, &wk_q, &lwork, &info));
  lwork = (PetscBLASInt)PetscRealPart(wk_q);
  PetscScalar *work;
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, A, &lda_, tau, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgeqrf LSQ: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* DORMQR: b := Q' * b   (side='L', trans='T'). */
  lwork = -1;
  PetscCallBLAS("LAPACKormqr", LAPACKormqr_("L", "T", &m_, &one, &n_, A, &lda_, tau, b, &m_, &wk_q, &lwork, &info));
  lwork = (PetscBLASInt)PetscRealPart(wk_q);
  PetscCall(PetscMalloc1(lwork, &work));
  PetscCallBLAS("LAPACKormqr", LAPACKormqr_("L", "T", &m_, &one, &n_, A, &lda_, tau, b, &m_, work, &lwork, &info));
  PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKormqr LSQ: info=%d", (int)info);
  PetscCall(PetscFree(work));

  /* DTRSV: solve R(0:n,0:n) * x = b(0:n) in place. R is in the upper
     triangle of A, ldA. */
  PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "N", &n_, A, &lda_, b, &one));

  PetscCall(PetscFree(tau));
  PetscFunctionReturn(PETSC_SUCCESS);
}
