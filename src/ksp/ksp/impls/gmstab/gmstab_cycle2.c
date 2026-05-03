/*
   GMstab2 — L=2 cycle. Direct port of gmstab_cpp/src/solver.cpp::gmstab2
   (lines 347-558). Each step is annotated with the corresponding C++
   line number so divergence between the ports can be localized.

   Algorithm summary (single cycle, m = 2s+2):

     1. (W, Y, H, Qh, Rh, x, r, beta) := aug_gmres_m(A, P, Z, V0, V1,
                                                      x, r, beta, m=2s+2)
                                          [counts 2s+2 matvecs]
     2. If beta < tolabs: return.
     3. gamma_vec = [beta; 0; ...; 0]   ((2s+1) × 1)
        Qy  = roworth(Y(:, 0..2s))     ; Qyh = roworth(Y(:, 0..2s+1))
        Slice H, Qh, Rh: H0 = H(0..2s, 0..2s-1) etc.
     4. Solve M_stack * inner = rhs_stack via tall LSQ (qr-based);
        xi = Rh0 \ inner.
     5. c0 = gamma_vec - H0*xi;  c1 = H1*c0;  c2 = H2*c1;  beta = ||c0||
     6. (tau1, tau2, beta) := StabCoeffs([c0, c1, c2], beta)
     7. x update: x += W*lin_term - V0*(Z\(Y(:,0..2s-1)*xi)).
     8. If beta <= tolabs: snapshot, return.
     9. BGS-extension of V1 against [W (full), V1_earlier], producing Rw.
     10. Qg = nullbasis(H0' * Qy')
     11. C_blk = term1 - tau1*term2 - tau2*term3
        F = Rw * C_blk * Qg
        [Qf, Rf] = qr(F, 0)
     12. QgRf = Qg / Rf;  tmp_lq = -tau2 * Y(:,0..2s+1) * H1 * (H0 * QgRf)
        [Qz, Lz_new] = lq(tmp_lq)
     13. big_combined = big_top - tau1*big_t1 - tau2*big_t2
        Rf_solve_Qz = Rf \ Qz
        coeffs = big_combined * (Qg * Rf_solve_Qz)
     14. V0 := [W(:,0..2s+1), V0] * coeffs
        V1 := [W(:,0..2s+2), V1] * (Qf * Qz)
        Z := Lz_new
     15. r := W * (c0_full - tau1*c1_full - tau2*c2_full)
     16. Snapshot.
*/
#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_internal.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_dump.h>
#include <math.h>

/* Snapshot callback for aug_gmres_m mid-iter snapshots, mirroring
   cycle1_snap_cb in gmstab_cycle1.c. */
typedef struct {
  KSP         ksp;
  KSP_GMSTAB *gms;
} cycle2_snap_ctx_t;

static PetscErrorCode cycle2_snap_cb(KSP ksp, Vec x_global_plus_local, PetscReal iter_norm,
                                      PetscBool *should_terminate, void *ctx_v)
{
  PetscFunctionBegin;
  cycle2_snap_ctx_t *ctx = (cycle2_snap_ctx_t *)ctx_v;
  PetscCall(KSPGMSTABSnapshot_Private(ctx->ksp, ctx->gms, x_global_plus_local, iter_norm));
  *should_terminate = (PetscBool)(ksp->reason != KSP_CONVERGED_ITERATING);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABCycle2_Private(KSP ksp, KSP_GMSTAB *gms,
                                                     KSPGMSTABInnerWorkspace *ws,
                                                     Vec x_local, Vec r0, PetscReal *beta_io)
{
  PetscFunctionBegin;
  const PetscInt s = gms->s;
  PetscReal      beta = *beta_io;

  /* ---- chk00: ENTRY ---------------------------------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private  (0, "V0_in",   gms->V0));
  PetscCall(KSPGMSTABDumpMat_Private  (0, "V1_in",   gms->V1));
  PetscCall(KSPGMSTABDumpDense_Private(0, "Z_in",    gms->Z, gms->Z_ldim, s, s));
  PetscCall(KSPGMSTABDumpVec_Private  (0, "x_in",    x_local));
  PetscCall(KSPGMSTABDumpVec_Private  (0, "r0_in",   r0));
  PetscCall(KSPGMSTABDumpReal_Private (0, "beta_in", beta));

  /* (1) aug_gmres_m for m = 2s+2 steps.
         C++ ref: solver.cpp:355-364 — aug_gmres_m(A, P, Z, V0, V1, x, r0, beta, m=2s+2). */
  const PetscInt m   = 2 * s + 2;
  const PetscInt mp1 = m + 1;
  ws->m = m;
  /* Reset inner-GMRES workspace state. The buffers are sized for
     m_max ≥ m, so setting only the active region is safe. */
  PetscCall(PetscArrayzero(ws->H,     (size_t)mp1 * (size_t)m));
  PetscCall(PetscArrayzero(ws->R,     (size_t)mp1 * (size_t)m));
  PetscCall(PetscArrayzero(ws->Q,     (size_t)mp1 * (size_t)mp1));
  PetscCall(PetscArrayzero(ws->Y,     (size_t)s   * (size_t)mp1));
  PetscCall(PetscArrayzero(ws->gamma, (size_t)mp1));
  PetscCall(PetscArrayzero(ws->G,     (size_t)m));
  for (PetscInt q = 0; q < mp1; ++q) ws->Q[(size_t)q + (size_t)q * (size_t)mp1] = 1.0;

  cycle2_snap_ctx_t snap_ctx = {ksp, gms};
  Vec x_p, r_p;
  PetscCall(VecDuplicate(x_local, &x_p));
  PetscCall(VecDuplicate(r0,      &r_p));
  PetscBool inner_converged = PETSC_FALSE;
  PetscReal beta_after_aug;
  PetscCall(KSPGMSTABAugGmresM_Private(ksp, gms->P, gms->Z, gms->Z_ldim,
            gms->V0, gms->V1,
            x_local, r0, beta,
            ksp->abstol, ws,
            gms->x_global, cycle2_snap_cb, &snap_ctx,
            x_p, r_p, &beta_after_aug, &inner_converged));

  /* Adopt outputs. */
  PetscCall(VecCopy(x_p, x_local));
  PetscCall(VecCopy(r_p, r0));
  beta = beta_after_aug;
  PetscCall(VecDestroy(&x_p));
  PetscCall(VecDestroy(&r_p));

  /* ---- chk01: AFTER aug_gmres_m --------------------------------------
     Note: aug_gmres_m's W is N × (m+1) = N × (2s+3); Y is s × (m+1)
     (PETSc allocates one column wider than C++'s s × m, but only fills
     the first m columns — the C++ dump uses Y's full m-column extent,
     so we slice s × m here). The H, Qh, Rh extractions use the same
     shapes the cycle2 algebra below needs. */
  {
    /* W: N × (m+1) — full slice */
    {
      const PetscScalar *Warr;
      PetscInt           ldW;
      PetscCall(KSPGMSTABDumpMatCols_Private(1, "W", ws->W, mp1));
      (void)Warr; (void)ldW;
    }
    /* Y: s × (m+1) — but C++ dumps s × m (only the columns aug_gmres
       fills). We dump matching m columns. */
    PetscCall(KSPGMSTABDumpDense_Private(1, "Y", ws->Y, s,        s,        m));
    /* H: (m+1) × m — matches C++'s H */
    PetscCall(KSPGMSTABDumpDense_Private(1, "H", ws->H, mp1,      mp1,      m));

    /* Qh extraction matches the C++ port's Q.topRows(m).transpose() —
       i.e. Qh[a, b] = ws->Q[b, a] for a ∈ [0, m+1), b ∈ [0, m). */
    PetscScalar *Qh_dump;
    PetscCall(PetscMalloc1((size_t)mp1 * (size_t)m, &Qh_dump));
    for (PetscInt b_ = 0; b_ < m; ++b_)
      for (PetscInt a_ = 0; a_ < mp1; ++a_)
        Qh_dump[(size_t)a_ + (size_t)b_ * (size_t)mp1] =
            ws->Q[(size_t)b_ + (size_t)a_ * (size_t)mp1];
    PetscCall(KSPGMSTABDumpDense_Private(1, "Qh", Qh_dump, mp1, mp1, m));
    PetscCall(PetscFree(Qh_dump));

    /* Rh: top m × m of ws->R (upper triangle only — but the C++ side
       dumps the full m × m block including zeros below the diagonal). */
    PetscScalar *Rh_dump;
    PetscCall(PetscCalloc1((size_t)m * (size_t)m, &Rh_dump));
    for (PetscInt b_ = 0; b_ < m; ++b_)
      for (PetscInt a_ = 0; a_ <= b_; ++a_)
        Rh_dump[(size_t)a_ + (size_t)b_ * (size_t)m] =
            ws->R[(size_t)a_ + (size_t)b_ * (size_t)mp1];
    PetscCall(KSPGMSTABDumpDense_Private(1, "Rh", Rh_dump, m, m, m));
    PetscCall(PetscFree(Rh_dump));

    PetscCall(KSPGMSTABDumpVec_Private (1, "x_postAug",    x_local));
    PetscCall(KSPGMSTABDumpVec_Private (1, "r_postAug",    r0));
    PetscCall(KSPGMSTABDumpReal_Private(1, "beta_postAug", beta));
  }

  /* (2) Convergence check. */
  if (beta < ksp->abstol) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* (3) gamma_vec, Qy, Qyh.
         C++ ref: solver.cpp:368-375.
         Qy  = roworth(Y[:, 0..2s])    → rows × (2s+1), where rows = rank(Y[:, 0..2s])
         Qyh = roworth(Y[:, 0..2s+1])  → rows × (2s+2)
         roworth(M) = orth(M')'.  For full-rank Y (rows = s), rows = s.
   */
  const PetscInt twos_p1 = 2 * s + 1;     /* number of cols in Qy's input  */
  const PetscInt twos_p2 = 2 * s + 2;     /* number of cols in Qyh's input */

  /* Helper inline: do roworth into the caller-allocated `Q_out`
     (assumed to have leading dim = rank_max = s). Rank reported via
     `*rank_out`. We pass the input's column count `nc` (so the input
     view is Y[:, 0..nc-1] of ws->Y). */
  /* For both Qy and Qyh, expected rank == s. Allocate accordingly. */
  PetscScalar *Yt;
  PetscScalar *Qbuf;
  PetscScalar *Qy;
  PetscScalar *Qyh;
  PetscInt     rank_y;
  PetscInt     rank_yh;
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)s, &Yt));
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)s, &Qbuf));
  PetscCall(PetscMalloc1((size_t)s * (size_t)twos_p1, &Qy));
  PetscCall(PetscMalloc1((size_t)s * (size_t)twos_p2, &Qyh));

  /* Build Yt = Y[:, 0..twos_p1-1]^T  (twos_p1 × s, ldim = twos_p1) */
  for (PetscInt i = 0; i < twos_p1; ++i)
    for (PetscInt j = 0; j < s; ++j)
      Yt[(size_t)i + (size_t)j * (size_t)twos_p1] =
          ws->Y[(size_t)j + (size_t)i * (size_t)s];
  /* orth(Yt): returns Q of shape twos_p1 × rank_y, ldim twos_p1 */
  PetscCall(KSPGMSTABOrth_Private(Yt, twos_p1, twos_p1, s, Qbuf, twos_p1, &rank_y));
  /* Transpose Qbuf (twos_p1 × rank_y) into Qy (rank_y × twos_p1) */
  for (PetscInt i = 0; i < rank_y; ++i)
    for (PetscInt j = 0; j < twos_p1; ++j)
      Qy[(size_t)i + (size_t)j * (size_t)rank_y] =
          Qbuf[(size_t)j + (size_t)i * (size_t)twos_p1];

  /* Same for Qyh — Y[:, 0..twos_p2-1]^T → twos_p2 × s */
  for (PetscInt i = 0; i < twos_p2; ++i)
    for (PetscInt j = 0; j < s; ++j)
      Yt[(size_t)i + (size_t)j * (size_t)twos_p2] =
          ws->Y[(size_t)j + (size_t)i * (size_t)s];
  PetscCall(KSPGMSTABOrth_Private(Yt, twos_p2, twos_p2, s, Qbuf, twos_p2, &rank_yh));
  for (PetscInt i = 0; i < rank_yh; ++i)
    for (PetscInt j = 0; j < twos_p2; ++j)
      Qyh[(size_t)i + (size_t)j * (size_t)rank_yh] =
          Qbuf[(size_t)j + (size_t)i * (size_t)twos_p2];

  /* ---- chk02: AFTER roworth -----------------------------------------
     Note: Qy and Qyh are SVD-derived row-orthonormal bases. Eigen
     BDCSVD and LAPACK DGESDD pick different sign conventions, so the
     dumps will DRIFT at order 1 between PETSc and C++. The downstream
     algebra absorbs this gauge: the least-squares solve `M_stack \
     rhs_stack` is invariant under sign-flips of M_stack rows that are
     mirrored in rhs_stack, so xi (chk04) onward through chk08 should
     match between ports at machine precision. The final outputs at
     chk16 are basis-invariant. Intermediate chks 09-15 will DRIFT
     because Qg = nullbasis(...) inherits the basis ambiguity. */
  PetscCall(KSPGMSTABDumpDense_Private(2, "Qy",  Qy,  rank_y,  rank_y,  twos_p1));
  PetscCall(KSPGMSTABDumpDense_Private(2, "Qyh", Qyh, rank_yh, rank_yh, twos_p2));

  /* (3) Build slices of H, Qh, Rh.
         C++ ref: solver.cpp:382-386.
         Working dimensions for s=4, m=2s+2=10:
           H0  : (2s+1) × 2s      = 9 × 8     — top-left of H (mp1 × m)
           Qh0 : (2s+1) × 2s      = 9 × 8     — top-left of Qh (mp1 × m)
           Rh0 : 2s × 2s          = 8 × 8     — top-left of Rh (m × m)
           H1  : (2s+2) × (2s+1)  = 10 × 9    — top-left of H
           H2  : (2s+3) × (2s+2)  = 11 × 10   — equals full H (mp1 × m)
   */
  const PetscInt twos = 2 * s;
  PetscScalar *H0;
  PetscScalar *Qh0_;
  PetscScalar *Rh0;
  PetscScalar *H1m;
  PetscScalar *H2m;
  PetscCall(PetscMalloc1((size_t)twos_p1 * (size_t)twos,    &H0));
  PetscCall(PetscMalloc1((size_t)twos_p1 * (size_t)twos,    &Qh0_));
  PetscCall(PetscMalloc1((size_t)twos    * (size_t)twos,    &Rh0));
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)twos_p1, &H1m));
  PetscCall(PetscMalloc1((size_t)mp1     * (size_t)m,       &H2m));
  /* H is stored as mp1 × m, ldim=mp1. Qh extracted from ws->Q as mp1 × m
     via the same transpose pattern as cycle 1. R is mp1 × m, ldim=mp1. */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p1; ++r_)
      H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p1] =
          ws->H[(size_t)r_ + (size_t)c_ * (size_t)mp1];
  /* Qh is the topRows(m).transpose() of ws->Q — i.e., Qh[a, b] = ws->Q[b, a].
     For the (2s+1) × (2s) Qh0 slice: a ∈ [0, 2s+1), b ∈ [0, 2s). */
  for (PetscInt b_ = 0; b_ < twos; ++b_)
    for (PetscInt a_ = 0; a_ < twos_p1; ++a_)
      Qh0_[(size_t)a_ + (size_t)b_ * (size_t)twos_p1] =
          ws->Q[(size_t)b_ + (size_t)a_ * (size_t)mp1];
  /* Rh0: top-left twos × twos block of upper-tri ws->R (mp1 × m, ldim=mp1).
     Below the diagonal the C++ port leaves zeros (Eigen Upper view); we
     match by zeroing those entries explicitly. */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos; ++r_) {
      const PetscScalar v = (r_ <= c_) ? ws->R[(size_t)r_ + (size_t)c_ * (size_t)mp1] : 0.0;
      Rh0[(size_t)r_ + (size_t)c_ * (size_t)twos] = v;
    }
  /* H1: (2s+2) × (2s+1) top-left of H (mp1 × m). */
  for (PetscInt c_ = 0; c_ < twos_p1; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p2; ++r_)
      H1m[(size_t)r_ + (size_t)c_ * (size_t)twos_p2] =
          ws->H[(size_t)r_ + (size_t)c_ * (size_t)mp1];
  /* H2: full mp1 × m of H (since 2s+3 = mp1, 2s+2 = m). */
  for (PetscInt c_ = 0; c_ < m; ++c_)
    for (PetscInt r_ = 0; r_ < mp1; ++r_)
      H2m[(size_t)r_ + (size_t)c_ * (size_t)mp1] =
          ws->H[(size_t)r_ + (size_t)c_ * (size_t)mp1];

  /* (4) gamma_vec = [beta; 0; ...; 0]   ((2s+1) × 1). */
  PetscScalar *gamma_vec;
  PetscCall(PetscCalloc1(twos_p1, &gamma_vec));
  gamma_vec[0] = (PetscScalar)beta;

  /* (5) M_stack = [Qy*Qh0; Qyh*(H1*Qh0)]    (rows_y + rows_yh) × 2s
        rhs_stack = [Qy*gamma; Qyh*(H1*gamma)]  (rows_y + rows_yh) × 1
        rows_y = rank_y  (typically s); rows_yh = rank_yh (typically s). */
  const PetscInt M_rows = rank_y + rank_yh;
  PetscScalar *M_stack;
  PetscScalar *rhs_stack;
  PetscScalar *top_blk, *bot_blk, *H1Qh0, *H1gamma;
  PetscCall(PetscMalloc1((size_t)M_rows * (size_t)twos, &M_stack));
  PetscCall(PetscMalloc1(M_rows, &rhs_stack));
  PetscCall(PetscMalloc1((size_t)rank_y  * (size_t)twos, &top_blk));
  PetscCall(PetscMalloc1((size_t)rank_yh * (size_t)twos, &bot_blk));
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)twos, &H1Qh0));
  PetscCall(PetscMalloc1(twos_p2, &H1gamma));

  /* H1Qh0 = H1 * Qh0  (twos_p2 × twos) */
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos_p2, twos, twos_p1, 1.0,
                                   H1m, twos_p2, Qh0_, twos_p1, 0.0, H1Qh0, twos_p2));
  /* top_blk = Qy * Qh0   (rank_y × twos)  — Qy is rank_y × twos_p1, Qh0 is twos_p1 × twos */
  PetscCall(KSPGMSTABGemm_Private("N", "N", rank_y, twos, twos_p1, 1.0,
                                   Qy, rank_y, Qh0_, twos_p1, 0.0, top_blk, rank_y));
  /* bot_blk = Qyh * H1Qh0  (rank_yh × twos)  — Qyh is rank_yh × twos_p2, H1Qh0 is twos_p2 × twos */
  PetscCall(KSPGMSTABGemm_Private("N", "N", rank_yh, twos, twos_p2, 1.0,
                                   Qyh, rank_yh, H1Qh0, twos_p2, 0.0, bot_blk, rank_yh));

  /* Pack into M_stack column-major (M_rows × twos). */
  for (PetscInt c_ = 0; c_ < twos; ++c_) {
    for (PetscInt r_ = 0; r_ < rank_y; ++r_)
      M_stack[(size_t)r_ + (size_t)c_ * (size_t)M_rows] =
          top_blk[(size_t)r_ + (size_t)c_ * (size_t)rank_y];
    for (PetscInt r_ = 0; r_ < rank_yh; ++r_)
      M_stack[(size_t)(rank_y + r_) + (size_t)c_ * (size_t)M_rows] =
          bot_blk[(size_t)r_ + (size_t)c_ * (size_t)rank_yh];
  }

  /* H1gamma = H1 * gamma_vec  (twos_p2 × 1) */
  PetscCall(KSPGMSTABGemv_Private("N", twos_p2, twos_p1, 1.0,
                                   H1m, twos_p2, gamma_vec, 1, 0.0, H1gamma, 1));
  /* rhs_top = Qy * gamma_vec   (rank_y × 1) */
  PetscCall(KSPGMSTABGemv_Private("N", rank_y, twos_p1, 1.0,
                                   Qy, rank_y, gamma_vec, 1, 0.0, rhs_stack, 1));
  /* rhs_bot = Qyh * H1gamma    (rank_yh × 1)   appended after rhs_top */
  PetscCall(KSPGMSTABGemv_Private("N", rank_yh, twos_p2, 1.0,
                                   Qyh, rank_yh, H1gamma, 1, 0.0, &rhs_stack[rank_y], 1));

  /* ---- chk03: AFTER M_stack / rhs_stack -----------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private  (3, "M_stack",   M_stack,   M_rows, M_rows, twos));
  PetscCall(KSPGMSTABDumpVector1d_Private(3, "rhs_stack", rhs_stack, M_rows));

  /* (6) Tall LSQ: solve M_stack * inner = rhs_stack via QR (Eigen ref).
         KSPGMSTABQrLeastSquares_Private mutates A and b in place;
         reads inner out of b's first n entries. */
  PetscScalar *Mcopy, *bcopy;
  PetscCall(PetscMalloc1((size_t)M_rows * (size_t)twos, &Mcopy));
  PetscCall(PetscMalloc1(M_rows, &bcopy));
  for (PetscInt q = 0; q < M_rows * twos; ++q) Mcopy[q] = M_stack[q];
  for (PetscInt q = 0; q < M_rows; ++q) bcopy[q] = rhs_stack[q];
  PetscCall(KSPGMSTABQrLeastSquares_Private(M_rows, twos, Mcopy, M_rows, bcopy));
  /* inner is now in bcopy[0..twos-1]. */
  PetscScalar *inner;
  PetscCall(PetscMalloc1(twos, &inner));
  for (PetscInt q = 0; q < twos; ++q) inner[q] = bcopy[q];
  PetscCall(PetscFree(Mcopy));
  PetscCall(PetscFree(bcopy));

  /* (7) xi = Rh0 \ inner   (upper-triangular trsv). */
  PetscScalar *xi;
  PetscCall(PetscMalloc1(twos, &xi));
  for (PetscInt q = 0; q < twos; ++q) xi[q] = inner[q];
  PetscCall(KSPGMSTABTrsv_Private("U", "N", "N", twos, Rh0, twos, xi, 1));

  /* ---- chk04: AFTER inner / xi --------------------------------------*/
  PetscCall(KSPGMSTABDumpVector1d_Private(4, "inner", inner, twos));
  PetscCall(KSPGMSTABDumpVector1d_Private(4, "xi",    xi,    twos));

  /* (8) c0 = gamma_vec - H0 * xi   ((2s+1) × 1)
         c1 = H1 * c0               ((2s+2) × 1)
         c2 = H2 * c1               ((2s+3) × 1) = (mp1) × 1
         beta = ||c0||
   */
  PetscScalar *c0_v;
  PetscScalar *c1_v;
  PetscScalar *c2_v;
  PetscCall(PetscMalloc1(twos_p1, &c0_v));
  PetscCall(PetscMalloc1(twos_p2, &c1_v));
  PetscCall(PetscMalloc1(mp1,     &c2_v));
  /* c0 := gamma_vec; then c0 -= H0 * xi */
  for (PetscInt q = 0; q < twos_p1; ++q) c0_v[q] = gamma_vec[q];
  PetscCall(KSPGMSTABGemv_Private("N", twos_p1, twos, -1.0,
                                   H0, twos_p1, xi, 1, 1.0, c0_v, 1));
  /* c1 = H1 * c0 */
  PetscCall(KSPGMSTABGemv_Private("N", twos_p2, twos_p1, 1.0,
                                   H1m, twos_p2, c0_v, 1, 0.0, c1_v, 1));
  /* c2 = H2 * c1   (H2 is mp1 × m with mp1 = 2s+3, m = 2s+2; c1 has 2s+2 entries) */
  PetscCall(KSPGMSTABGemv_Private("N", mp1, m, 1.0,
                                   H2m, mp1, c1_v, 1, 0.0, c2_v, 1));
  {
    PetscReal cn = 0.0;
    for (PetscInt q = 0; q < twos_p1; ++q) cn += PetscRealPart(c0_v[q]) * PetscRealPart(c0_v[q]);
    beta = PetscSqrtReal(cn);
  }

  /* ---- chk05: c0, c1, c2 --------------------------------------------*/
  PetscCall(KSPGMSTABDumpVector1d_Private(5, "c0",         c0_v, twos_p1));
  PetscCall(KSPGMSTABDumpVector1d_Private(5, "c1",         c1_v, twos_p2));
  PetscCall(KSPGMSTABDumpVector1d_Private(5, "c2",         c2_v, mp1));
  PetscCall(KSPGMSTABDumpReal_Private    (5, "beta_postC", beta));

  /* (9) StabCoeffs(L=2, [c0_padded, c1_padded, c2], beta) → tau1, tau2, beta_new.
         C++ ref: solver.cpp:413-422.
         We need to wrap c0_v, c1_v, c2_v into three sequential PETSc
         Vecs of common length mp1 = 2s+3. c0 (length 2s+1) and c1
         (length 2s+2) are zero-padded to mp1. */
  Vec stab_v0, stab_v1, stab_v2;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, mp1, &stab_v0));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, mp1, &stab_v1));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, mp1, &stab_v2));
  PetscCall(VecSet(stab_v0, 0.0));
  PetscCall(VecSet(stab_v1, 0.0));
  PetscCall(VecSet(stab_v2, 0.0));
  {
    PetscScalar *p0, *p1, *p2;
    PetscCall(VecGetArray(stab_v0, &p0));
    PetscCall(VecGetArray(stab_v1, &p1));
    PetscCall(VecGetArray(stab_v2, &p2));
    for (PetscInt q = 0; q < twos_p1; ++q) p0[q] = c0_v[q];
    for (PetscInt q = 0; q < twos_p2; ++q) p1[q] = c1_v[q];
    for (PetscInt q = 0; q < mp1;     ++q) p2[q] = c2_v[q];
    PetscCall(VecRestoreArray(stab_v0, &p0));
    PetscCall(VecRestoreArray(stab_v1, &p1));
    PetscCall(VecRestoreArray(stab_v2, &p2));
  }
  Vec stab_cols[3] = {stab_v0, stab_v1, stab_v2};
  PetscScalar tau_buf[2];
  PetscReal beta_after_stab2;
  PetscCall(KSPGMSTABStabCoeffs_Private(stab_cols, /*L=*/2, beta, gms->stab_angle, tau_buf, &beta_after_stab2));
  const PetscScalar tau1 = tau_buf[0];
  const PetscScalar tau2 = tau_buf[1];
  beta = beta_after_stab2;
  PetscCall(VecDestroy(&stab_v0));
  PetscCall(VecDestroy(&stab_v1));
  PetscCall(VecDestroy(&stab_v2));

  /* ---- chk06: AFTER StabCoeffs --------------------------------------*/
  PetscCall(KSPGMSTABDumpReal_Private(6, "tau1",     (PetscReal)PetscRealPart(tau1)));
  PetscCall(KSPGMSTABDumpReal_Private(6, "tau2",     (PetscReal)PetscRealPart(tau2)));
  PetscCall(KSPGMSTABDumpReal_Private(6, "beta_new", beta));

  /* (10) x update.
          C++ ref: solver.cpp:425-434.
          xi_pad      = [xi(0..2s-1); 0; 0]      length mp1 (with last 2 zero)
                       — but in the C++ code, xi_pad has length 2s+2 (m).
          c0_pad      = [c0(0..2s); 0]            length 2s+2 (with last zero)
          lin_term    = xi_pad + tau1*c0_pad + tau2*c1
                       (length 2s+2 = m)
          x += W(:, 0..2s+1) * lin_term  - V0 * (Z \ (Y(:,0..2s-1) * xi))
   */
  PetscScalar *xi_pad;
  PetscScalar *c0_pad;
  PetscScalar *lin_term;
  PetscCall(PetscCalloc1(m, &xi_pad));   /* length 2s+2 */
  PetscCall(PetscCalloc1(m, &c0_pad));   /* length 2s+2 */
  PetscCall(PetscMalloc1(m, &lin_term));
  for (PetscInt q = 0; q < twos;    ++q) xi_pad[q] = xi[q];        /* first 2s */
  for (PetscInt q = 0; q < twos_p1; ++q) c0_pad[q] = c0_v[q];      /* first 2s+1 */
  for (PetscInt q = 0; q < m; ++q) lin_term[q] = xi_pad[q] + tau1 * c0_pad[q] + tau2 * c1_v[q];

  /* Yxi = Y(:, 0..2s-1) * xi   (length s) */
  PetscScalar *Yxi;
  PetscScalar *Zinv_Yxi;
  PetscCall(PetscMalloc1(s, &Yxi));
  PetscCall(PetscMalloc1(s, &Zinv_Yxi));
  PetscCall(KSPGMSTABGemv_Private("N", s, twos, 1.0, ws->Y, s, xi, 1, 0.0, Yxi, 1));
  for (PetscInt q = 0; q < s; ++q) Zinv_Yxi[q] = Yxi[q];
  PetscCall(KSPGMSTABTrsv_Private("L", "N", "N", s, gms->Z, gms->Z_ldim, Zinv_Yxi, 1));

  /* x += W(:, 0..2s+1) * lin_term  (parallel mat-vec via column-by-column AXPY) */
  for (PetscInt q = 0; q < m; ++q) {
    Vec wq;
    PetscCall(MatDenseGetColumnVecRead(ws->W, q, &wq));
    PetscCall(VecAXPY(x_local, lin_term[q], wq));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, q, &wq));
  }
  /* x -= V0(:, k) * Zinv_Yxi[k]   for k = 0..s-1 */
  for (PetscInt k = 0; k < s; ++k) {
    Vec V0k;
    PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
    PetscCall(VecAXPY(x_local, -Zinv_Yxi[k], V0k));
    PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
  }

  /* ---- chk07: AFTER x update ----------------------------------------*/
  PetscCall(KSPGMSTABDumpVec_Private(7, "x_postUpd", x_local));

  /* (11) Mid-cycle snapshot (matches C++ solver.cpp:436-438). */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if (beta <= ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING)) {
    /* Early-out cleanup. */
    PetscCall(PetscFree(Zinv_Yxi));
    PetscCall(PetscFree(Yxi));
    PetscCall(PetscFree(lin_term));
    PetscCall(PetscFree(c0_pad));
    PetscCall(PetscFree(xi_pad));
    PetscCall(PetscFree(c2_v));
    PetscCall(PetscFree(c1_v));
    PetscCall(PetscFree(c0_v));
    PetscCall(PetscFree(xi));
    PetscCall(PetscFree(inner));
    PetscCall(PetscFree(H1gamma));
    PetscCall(PetscFree(H1Qh0));
    PetscCall(PetscFree(bot_blk));
    PetscCall(PetscFree(top_blk));
    PetscCall(PetscFree(rhs_stack));
    PetscCall(PetscFree(M_stack));
    PetscCall(PetscFree(gamma_vec));
    PetscCall(PetscFree(H2m));
    PetscCall(PetscFree(H1m));
    PetscCall(PetscFree(Rh0));
    PetscCall(PetscFree(Qh0_));
    PetscCall(PetscFree(H0));
    PetscCall(PetscFree(Qy));
    PetscCall(PetscFree(Qyh));
    PetscCall(PetscFree(Qbuf));
    PetscCall(PetscFree(Yt));
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* (12) BGS-extension of V1 against [W (first 2s+3 cols), V1_earlier].
          C++ ref: solver.cpp:441-458.
          Rw is (3s+3) × (3s+3) with:
            Rw[0:2s+3, 0:2s+3] = I_{2s+3}
            Rw[j, 2s+3+i]      = <W(:, j), V1(:, i)>           j < 2s+3, i ∈ [0, s)
            Rw[2s+3+j, 2s+3+i] = <V1(:, j)_new, V1(:, i)_part>  j < i
            Rw[2s+3+i, 2s+3+i] = ||V1(:, i)||_post
   */
  const PetscInt twos_p3 = 2 * s + 3;
  const PetscInt Rw_size = 3 * s + 3;
  PetscScalar *Rw;
  PetscCall(PetscCalloc1((size_t)Rw_size * (size_t)Rw_size, &Rw));
  /* Top-left identity: Rw[i, i] = 1 for i = 0..2s+2 */
  for (PetscInt q = 0; q < twos_p3; ++q) Rw[(size_t)q + (size_t)q * (size_t)Rw_size] = 1.0;

  for (PetscInt i = 0; i < s; ++i) {
    /* Get V1 column i (write access). */
    Vec V1i;
    PetscCall(MatDenseGetColumnVec(gms->V1, i, &V1i));
    /* Orthogonalize V1(:, i) against W(:, j) for j = 0..2s+2 */
    for (PetscInt j = 0; j < twos_p3; ++j) {
      Vec Wj;
      PetscScalar c_ji;
      PetscCall(MatDenseGetColumnVecRead(ws->W, j, &Wj));
      PetscCall(VecDot(V1i, Wj, &c_ji));
      PetscCall(VecAXPY(V1i, -c_ji, Wj));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, j, &Wj));
      Rw[(size_t)j + (size_t)(twos_p3 + i) * (size_t)Rw_size] = c_ji;
    }
    /* Release V1i before re-borrowing for the V0/V1 inner loop (PETSc
       MatDense double-borrow rule mirrors cycle 1's approach). */
    PetscCall(MatDenseRestoreColumnVec(gms->V1, i, &V1i));
    /* Copy V1(:, i) (the version after W-orthogonalization) into work_out
       to avoid double-borrowing during the inner V1-against-V1 loop. */
    PetscCall(MatDenseGetColumnVecRead(gms->V1, i, &V1i));
    PetscCall(VecCopy(V1i, ws->work_out));
    PetscCall(MatDenseRestoreColumnVecRead(gms->V1, i, &V1i));

    for (PetscInt j = 0; j < i; ++j) {
      Vec V1j;
      PetscScalar c_sji;
      PetscCall(MatDenseGetColumnVecRead(gms->V1, j, &V1j));
      PetscCall(VecDot(ws->work_out, V1j, &c_sji));
      PetscCall(VecAXPY(ws->work_out, -c_sji, V1j));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V1, j, &V1j));
      Rw[(size_t)(twos_p3 + j) + (size_t)(twos_p3 + i) * (size_t)Rw_size] = c_sji;
    }
    PetscReal nrm;
    PetscCall(VecNorm(ws->work_out, NORM_2, &nrm));
    Rw[(size_t)(twos_p3 + i) + (size_t)(twos_p3 + i) * (size_t)Rw_size] = (PetscScalar)nrm;
    PetscCall(VecScale(ws->work_out, 1.0 / nrm));
    /* Write back into V1[:, i]. */
    PetscCall(MatDenseGetColumnVec(gms->V1, i, &V1i));
    PetscCall(VecCopy(ws->work_out, V1i));
    PetscCall(MatDenseRestoreColumnVec(gms->V1, i, &V1i));
  }

  /* ---- chk08: AFTER BGS-extension into V1 ---------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private  (8, "V1_postBGS", gms->V1));
  PetscCall(KSPGMSTABDumpDense_Private(8, "Rw",         Rw, Rw_size, Rw_size, Rw_size));

  /* (13) Qg = nullbasis(H0^T * Qy^T).
          C++ ref: solver.cpp:461.
          H0 is (2s+1) × 2s; Qy is rank_y × (2s+1).
          H0^T * Qy^T = (2s × (2s+1)) × ((2s+1) × rank_y) = 2s × rank_y.
          nullbasis(M) for tall M (rows > cols) returns rows × (rows-cols).
          For full-rank Y (rank_y = s), Qg is 2s × (2s − s) = 2s × s. */
  PetscScalar *H0t_Qyt;  /* 2s × rank_y */
  PetscCall(PetscMalloc1((size_t)twos * (size_t)rank_y, &H0t_Qyt));
  /* Compute H0^T * Qy^T:
       result[i, j] = sum_k H0[k, i] * Qy[j, k]   for i ∈ [0, 2s), j ∈ [0, rank_y).
     Use GEMM with transA='T', transB='T', m=2s, n=rank_y, k=2s+1. */
  PetscCall(KSPGMSTABGemm_Private("T", "T", twos, rank_y, twos_p1, 1.0,
                                   H0, twos_p1, Qy, rank_y, 0.0, H0t_Qyt, twos));
  PetscScalar *Qg;
  const PetscInt Qg_cols = twos - rank_y;   /* expected = s when rank_y = s */
  /* The downstream algebra (lq, V0/V1 update, Z assignment) is sized for
     Qg_cols == s. If Y is rank-deficient (rank_y < s), Qg_cols > s and
     we'd overrun fixed-size buffers. Y is the projection P^T*W of the
     parallel Krylov basis through the shadow space; for a non-degenerate
     P and well-conditioned A it is full-rank by construction. Fail
     loudly if that assumption breaks. */
  PetscCheck(Qg_cols == s, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP,
             "KSPGMSTABCycle2_Private: nullbasis(H0' Qy') has %" PetscInt_FMT
             " columns, expected s=%" PetscInt_FMT ". This means Y was rank-deficient "
             "(rank_y=%" PetscInt_FMT " instead of s). Cycle 2 cannot proceed.",
             Qg_cols, s, rank_y);
  PetscCall(PetscMalloc1((size_t)twos * (size_t)Qg_cols, &Qg));
  PetscCall(KSPGMSTABNullBasis_Private(H0t_Qyt, twos, twos, rank_y, Qg, twos));

  /* ---- chk09: AFTER nullbasis ---------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(9, "Qg", Qg, twos, twos, Qg_cols));

  /* (14) Build term1, term2, term3, then C_blk.
          C++ ref: solver.cpp:473-494. */
  const PetscInt three_p3 = 3 * s + 3;
  PetscScalar *term1, *term2, *term3, *C_blk;
  PetscCall(PetscCalloc1((size_t)three_p3 * (size_t)twos, &term1));
  PetscCall(PetscCalloc1((size_t)three_p3 * (size_t)twos, &term2));
  PetscCall(PetscCalloc1((size_t)three_p3 * (size_t)twos, &term3));
  PetscCall(PetscMalloc1((size_t)three_p3 * (size_t)twos, &C_blk));

  /* term1: top (2s+1) rows = H0 */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p1; ++r_)
      term1[(size_t)r_ + (size_t)c_ * (size_t)three_p3] =
          H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p1];

  /* H1_H0 = H1 * H0   ((2s+2) × 2s) */
  PetscScalar *H1_H0;
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)twos, &H1_H0));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos_p2, twos, twos_p1, 1.0,
                                   H1m, twos_p2, H0, twos_p1, 0.0, H1_H0, twos_p2));

  /* term2: top (2s+2) rows = H1_H0 */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p2; ++r_)
      term2[(size_t)r_ + (size_t)c_ * (size_t)three_p3] =
          H1_H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p2];

  /* H2_H1_H0 = H2 * H1_H0   ((2s+3) × 2s) */
  PetscScalar *H2_H1_H0;
  PetscCall(PetscMalloc1((size_t)twos_p3 * (size_t)twos, &H2_H1_H0));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos_p3, twos, twos_p2, 1.0,
                                   H2m, mp1, H1_H0, twos_p2, 0.0, H2_H1_H0, twos_p3));

  /* Y_H1H0 = Y(:, 0..2s+1) * H1_H0   (s × 2s) */
  PetscScalar *Y_H1H0;
  PetscCall(PetscMalloc1((size_t)s * (size_t)twos, &Y_H1H0));
  PetscCall(KSPGMSTABGemm_Private("N", "N", s, twos, twos_p2, 1.0,
                                   ws->Y, s, H1_H0, twos_p2, 0.0, Y_H1H0, s));
  /* Zsolve_YH1H0 = Z \ Y_H1H0   — lower-tri solve from left (s × 2s) */
  PetscScalar *Zsolve_YH1H0;
  PetscCall(PetscMalloc1((size_t)s * (size_t)twos, &Zsolve_YH1H0));
  for (PetscInt q = 0; q < s * twos; ++q) Zsolve_YH1H0[q] = Y_H1H0[q];
  PetscCall(KSPGMSTABTrsm_Private("L", "L", "N", "N", s, twos, 1.0, gms->Z, gms->Z_ldim, Zsolve_YH1H0, s));

  /* term3: top (2s+3) rows = H2_H1_H0; bottom s rows = Zsolve_YH1H0 */
  for (PetscInt c_ = 0; c_ < twos; ++c_) {
    for (PetscInt r_ = 0; r_ < twos_p3; ++r_)
      term3[(size_t)r_ + (size_t)c_ * (size_t)three_p3] =
          H2_H1_H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p3];
    for (PetscInt r_ = 0; r_ < s; ++r_)
      term3[(size_t)(twos_p3 + r_) + (size_t)c_ * (size_t)three_p3] =
          Zsolve_YH1H0[(size_t)r_ + (size_t)c_ * (size_t)s];
  }

  /* C_blk = term1 - tau1 * term2 - tau2 * term3 */
  for (PetscInt q = 0; q < three_p3 * twos; ++q)
    C_blk[q] = term1[q] - tau1 * term2[q] - tau2 * term3[q];

  /* ---- chk10: AFTER C_blk -------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(10, "term1", term1, three_p3, three_p3, twos));
  PetscCall(KSPGMSTABDumpDense_Private(10, "term2", term2, three_p3, three_p3, twos));
  PetscCall(KSPGMSTABDumpDense_Private(10, "term3", term3, three_p3, three_p3, twos));
  PetscCall(KSPGMSTABDumpDense_Private(10, "C_blk", C_blk, three_p3, three_p3, twos));

  /* (15) F = Rw * C_blk * Qg     ((3s+3) × s) */
  PetscScalar *RwC, *F;
  PetscCall(PetscMalloc1((size_t)three_p3 * (size_t)twos,    &RwC));
  PetscCall(PetscMalloc1((size_t)three_p3 * (size_t)Qg_cols, &F));
  PetscCall(KSPGMSTABGemm_Private("N", "N", three_p3, twos, three_p3, 1.0,
                                   Rw, Rw_size, C_blk, three_p3, 0.0, RwC, three_p3));
  PetscCall(KSPGMSTABGemm_Private("N", "N", three_p3, Qg_cols, twos, 1.0,
                                   RwC, three_p3, Qg, twos, 0.0, F, three_p3));

  /* ---- chk11: AFTER F ----------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(11, "F", F, three_p3, three_p3, Qg_cols));

  /* (16) QR(F): thin Qf (3s+3) × Qg_cols, Rf (Qg_cols × Qg_cols).
          Same pattern as cycle 1's QR(F): dgeqrf + dorgqr. */
  PetscScalar *Qf, *Rf;
  PetscCall(PetscCalloc2((size_t)three_p3 * (size_t)Qg_cols, &Qf,
                          (size_t)Qg_cols * (size_t)Qg_cols, &Rf));
  {
    PetscScalar *Fbuf;
    PetscCall(PetscMalloc1((size_t)three_p3 * (size_t)Qg_cols, &Fbuf));
    for (PetscInt q = 0; q < three_p3 * Qg_cols; ++q) Fbuf[q] = F[q];

    PetscBLASInt m_, n_, lda_ = (PetscBLASInt)three_p3, lwork, info, k_ = (PetscBLASInt)Qg_cols;
    PetscCall(PetscBLASIntCast(three_p3, &m_));
    PetscCall(PetscBLASIntCast(Qg_cols,  &n_));

    PetscScalar *tau_v, wkq;
    PetscCall(PetscMalloc1(Qg_cols, &tau_v));
    lwork = -1;
    PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Fbuf, &lda_, tau_v, &wkq, &lwork, &info));
    lwork = (PetscBLASInt)PetscRealPart(wkq);
    PetscScalar *work;
    PetscCall(PetscMalloc1(lwork, &work));
    PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Fbuf, &lda_, tau_v, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "geqrf F (cycle2): info=%d", (int)info);
    PetscCall(PetscFree(work));

    /* Extract Rf (top Qg_cols × Qg_cols of factored F). */
    for (PetscInt c_ = 0; c_ < Qg_cols; ++c_)
      for (PetscInt r_ = 0; r_ <= c_; ++r_)
        Rf[(size_t)r_ + (size_t)c_ * (size_t)Qg_cols] =
            Fbuf[(size_t)r_ + (size_t)c_ * (size_t)three_p3];

    /* Reify thin Qf via dorgqr. */
    lwork = -1;
    PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &n_, &k_, Fbuf, &lda_, tau_v, &wkq, &lwork, &info));
    lwork = (PetscBLASInt)PetscRealPart(wkq);
    PetscCall(PetscMalloc1(lwork, &work));
    PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &n_, &k_, Fbuf, &lda_, tau_v, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "orgqr F (cycle2): info=%d", (int)info);
    PetscCall(PetscFree(work));

    for (PetscInt q = 0; q < three_p3 * Qg_cols; ++q) Qf[q] = Fbuf[q];
    PetscCall(PetscFree(tau_v));
    PetscCall(PetscFree(Fbuf));
  }

  /* ---- chk12: AFTER QR(F) ------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(12, "Qf", Qf, three_p3, three_p3, Qg_cols));
  PetscCall(KSPGMSTABDumpDense_Private(12, "Rf", Rf, Qg_cols, Qg_cols, Qg_cols));

  /* (17) QgRf = Qg / Rf   (= Qg * Rf^{-1}, 2s × s) via right-side trsm.
          tmp_lq = -tau2 * Y(:, 0..2s+1) * H1 * (H0 * QgRf)   (s × s) */
  PetscScalar *QgRf;
  PetscCall(PetscMalloc1((size_t)twos * (size_t)Qg_cols, &QgRf));
  for (PetscInt q = 0; q < twos * Qg_cols; ++q) QgRf[q] = Qg[q];
  PetscCall(KSPGMSTABTrsm_Private("R", "U", "N", "N", twos, Qg_cols, 1.0, Rf, Qg_cols, QgRf, twos));

  /* H0_QgRf = H0 * QgRf   ((2s+1) × Qg_cols) */
  PetscScalar *H0_QgRf;
  PetscCall(PetscMalloc1((size_t)twos_p1 * (size_t)Qg_cols, &H0_QgRf));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos_p1, Qg_cols, twos, 1.0,
                                   H0, twos_p1, QgRf, twos, 0.0, H0_QgRf, twos_p1));
  /* H1_H0_QgRf = H1 * H0_QgRf   ((2s+2) × Qg_cols) */
  PetscScalar *H1_H0_QgRf;
  PetscCall(PetscMalloc1((size_t)twos_p2 * (size_t)Qg_cols, &H1_H0_QgRf));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos_p2, Qg_cols, twos_p1, 1.0,
                                   H1m, twos_p2, H0_QgRf, twos_p1, 0.0, H1_H0_QgRf, twos_p2));
  /* Y_H1_H0_QgRf = Y(:, 0..2s+1) * H1_H0_QgRf   (s × Qg_cols) */
  PetscScalar *tmp_lq;
  PetscCall(PetscMalloc1((size_t)s * (size_t)Qg_cols, &tmp_lq));
  PetscCall(KSPGMSTABGemm_Private("N", "N", s, Qg_cols, twos_p2, -tau2,
                                   ws->Y, s, H1_H0_QgRf, twos_p2, 0.0, tmp_lq, s));

  /* ---- chk13: QgRf and tmp_lq ---------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(13, "QgRf",   QgRf,   twos, twos, Qg_cols));
  PetscCall(KSPGMSTABDumpDense_Private(13, "tmp_lq", tmp_lq, s,    s,    Qg_cols));

  /* (18) lq(tmp_lq) → Qz, Lz_new   (each s × s). */
  PetscScalar *Qz, *Lz_new;
  PetscCall(PetscMalloc2((size_t)s * (size_t)s, &Qz, (size_t)s * (size_t)s, &Lz_new));
  PetscCall(KSPGMSTABLq_Private(tmp_lq, s, s, Qg_cols, Qz, s, Lz_new, s));

  /* ---- chk14: AFTER lq(tmp_lq) -------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(14, "Qz",     Qz,     s, s, s));
  PetscCall(KSPGMSTABDumpDense_Private(14, "Lz_new", Lz_new, s, s, s));

  /* (19) big_combined construction.
          C++ ref: solver.cpp:521-535.
          big_top:        (3s+2) × 2s
            top (2s+2) rows: I_{2s+2, 2s}  (identity-padded, last 2 rows zero)
            bottom  s rows : -Z\Y(:, 0..2s-1)   (s × 2s)
          big_t1:         (3s+2) × 2s
            top (2s+1) rows: H0
            rest zeros
          big_t2:         (3s+2) × 2s
            top (2s+2) rows: H1*H0
            rest zeros
          big_combined = big_top - tau1*big_t1 - tau2*big_t2
   */
  const PetscInt big_rows = 3 * s + 2;
  PetscScalar *big_top, *big_t1, *big_t2, *big_combined;
  PetscCall(PetscCalloc1((size_t)big_rows * (size_t)twos, &big_top));
  PetscCall(PetscCalloc1((size_t)big_rows * (size_t)twos, &big_t1));
  PetscCall(PetscCalloc1((size_t)big_rows * (size_t)twos, &big_t2));
  PetscCall(PetscMalloc1((size_t)big_rows * (size_t)twos, &big_combined));

  /* big_top top: I_{2s+2, 2s}  (entry (i, j) = 1 iff i == j and i < 2s; else 0) */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    big_top[(size_t)c_ + (size_t)c_ * (size_t)big_rows] = 1.0;
  /* big_top bottom (rows 2s+2..3s+1): -Z \ Y(:, 0..2s-1) */
  PetscScalar *Zsolve_Ysmall;
  PetscCall(PetscMalloc1((size_t)s * (size_t)twos, &Zsolve_Ysmall));
  /* Copy Y(:, 0..2s-1) (s × 2s) into Zsolve_Ysmall */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < s; ++r_)
      Zsolve_Ysmall[(size_t)r_ + (size_t)c_ * (size_t)s] =
          ws->Y[(size_t)r_ + (size_t)c_ * (size_t)s];
  PetscCall(KSPGMSTABTrsm_Private("L", "L", "N", "N", s, twos, 1.0, gms->Z, gms->Z_ldim, Zsolve_Ysmall, s));
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < s; ++r_)
      big_top[(size_t)(twos_p2 + r_) + (size_t)c_ * (size_t)big_rows] =
          -Zsolve_Ysmall[(size_t)r_ + (size_t)c_ * (size_t)s];
  PetscCall(PetscFree(Zsolve_Ysmall));

  /* big_t1 top (2s+1) rows: H0 */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p1; ++r_)
      big_t1[(size_t)r_ + (size_t)c_ * (size_t)big_rows] =
          H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p1];

  /* big_t2 top (2s+2) rows: H1*H0 */
  for (PetscInt c_ = 0; c_ < twos; ++c_)
    for (PetscInt r_ = 0; r_ < twos_p2; ++r_)
      big_t2[(size_t)r_ + (size_t)c_ * (size_t)big_rows] =
          H1_H0[(size_t)r_ + (size_t)c_ * (size_t)twos_p2];

  /* big_combined = big_top - tau1*big_t1 - tau2*big_t2 */
  for (PetscInt q = 0; q < big_rows * twos; ++q)
    big_combined[q] = big_top[q] - tau1 * big_t1[q] - tau2 * big_t2[q];

  /* (20) Rf_solve_Qz = Rf \ Qz  (s × s, upper-tri solve from left)
         coeffs       = big_combined * (Qg * Rf_solve_Qz)   (big_rows × s) */
  PetscScalar *Rf_solve_Qz;
  PetscCall(PetscMalloc1((size_t)Qg_cols * (size_t)Qg_cols, &Rf_solve_Qz));
  for (PetscInt q = 0; q < Qg_cols * Qg_cols; ++q) Rf_solve_Qz[q] = Qz[q];
  PetscCall(KSPGMSTABTrsm_Private("L", "U", "N", "N", Qg_cols, Qg_cols, 1.0, Rf, Qg_cols, Rf_solve_Qz, Qg_cols));

  PetscScalar *Qg_Rf_Qz;
  PetscCall(PetscMalloc1((size_t)twos * (size_t)Qg_cols, &Qg_Rf_Qz));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos, Qg_cols, Qg_cols, 1.0,
                                   Qg, twos, Rf_solve_Qz, Qg_cols, 0.0, Qg_Rf_Qz, twos));

  PetscScalar *coeffs;
  PetscCall(PetscMalloc1((size_t)big_rows * (size_t)Qg_cols, &coeffs));
  PetscCall(KSPGMSTABGemm_Private("N", "N", big_rows, Qg_cols, twos, 1.0,
                                   big_combined, big_rows, Qg_Rf_Qz, twos, 0.0, coeffs, big_rows));

  /* ---- chk15: big_combined / Rf_solve_Qz / coeffs ------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(15, "big_combined", big_combined, big_rows, big_rows, twos));
  PetscCall(KSPGMSTABDumpDense_Private(15, "Rf_solve_Qz",  Rf_solve_Qz,  Qg_cols,  Qg_cols,  Qg_cols));
  PetscCall(KSPGMSTABDumpDense_Private(15, "coeffs",       coeffs,       big_rows, big_rows, Qg_cols));

  /* (21) V0 = [W(:, 0..2s+1), V0_old] * coeffs    (parallel mat * coeffs).
          big_rows = 2s+2 + s = 3s+2.
          For each output column c_ ∈ [0, Qg_cols):
            V0_new[:, c_] = sum_{k=0..2s+1} W[:, k] * coeffs[k, c_]
                          + sum_{k=0..s-1}  V0_old[:, k] * coeffs[2s+2+k, c_]
   */
  Mat V0_post;
  {
    PetscInt n_local;
    PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &V0_post));
    PetscCall(MatSetType(V0_post, MATDENSE));
    PetscCall(MatSetSizes(V0_post, n_local, PETSC_DECIDE, N_global, Qg_cols));
    PetscCall(MatSetUp(V0_post));
  }
  for (PetscInt c_ = 0; c_ < Qg_cols; ++c_) {
    Vec V0pc;
    PetscCall(MatDenseGetColumnVec(V0_post, c_, &V0pc));
    PetscCall(VecSet(V0pc, 0.0));
    /* W(:, 0..2s+1) contribution */
    for (PetscInt k = 0; k < twos_p2; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V0pc, coeffs[(size_t)k + (size_t)c_ * (size_t)big_rows], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    /* V0_old(:, 0..s-1) contribution */
    for (PetscInt k = 0; k < s; ++k) {
      Vec V0k;
      PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
      PetscCall(VecAXPY(V0pc, coeffs[(size_t)(twos_p2 + k) + (size_t)c_ * (size_t)big_rows], V0k));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
    }
    PetscCall(MatDenseRestoreColumnVec(V0_post, c_, &V0pc));
  }
  PetscCall(MatDestroy(&gms->V0));
  gms->V0 = V0_post;

  /* (22) V1 = [W(:, 0..2s+2), V1_old] * (Qf * Qz)
          QfQz = Qf * Qz  ((3s+3) × s).
          Note: cycle 2 has V1's Q product Q_z, distinct from cycle1. */
  PetscScalar *QfQz;
  PetscCall(PetscMalloc1((size_t)three_p3 * (size_t)Qg_cols, &QfQz));
  PetscCall(KSPGMSTABGemm_Private("N", "N", three_p3, Qg_cols, Qg_cols, 1.0,
                                   Qf, three_p3, Qz, s, 0.0, QfQz, three_p3));

  Mat V1_post;
  {
    PetscInt n_local;
    PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &V1_post));
    PetscCall(MatSetType(V1_post, MATDENSE));
    PetscCall(MatSetSizes(V1_post, n_local, PETSC_DECIDE, N_global, Qg_cols));
    PetscCall(MatSetUp(V1_post));
  }
  for (PetscInt c_ = 0; c_ < Qg_cols; ++c_) {
    Vec V1pc;
    PetscCall(MatDenseGetColumnVec(V1_post, c_, &V1pc));
    PetscCall(VecSet(V1pc, 0.0));
    for (PetscInt k = 0; k < twos_p3; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V1pc, QfQz[(size_t)k + (size_t)c_ * (size_t)three_p3], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    for (PetscInt k = 0; k < s; ++k) {
      Vec V1k;
      PetscCall(MatDenseGetColumnVecRead(gms->V1, k, &V1k));
      PetscCall(VecAXPY(V1pc, QfQz[(size_t)(twos_p3 + k) + (size_t)c_ * (size_t)three_p3], V1k));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V1, k, &V1k));
    }
    PetscCall(MatDenseRestoreColumnVec(V1_post, c_, &V1pc));
  }
  PetscCall(MatDestroy(&gms->V1));
  gms->V1 = V1_post;

  /* (23) Z := Lz_new */
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt r_ = 0; r_ < s; ++r_)
      gms->Z[(size_t)r_ + (size_t)c_ * (size_t)gms->Z_ldim] =
          Lz_new[(size_t)r_ + (size_t)c_ * (size_t)s];

  /* (24) r0 = W * (c0_full - tau1*c1_full - tau2*c2_full)
          where c0_full, c1_full, c2_full are zero-padded to length mp1 (= W.cols).
          Note: c0_v has length 2s+1, c1_v has 2s+2, c2_v has mp1 = 2s+3. */
  PetscScalar *r_coeffs;
  PetscCall(PetscCalloc1(mp1, &r_coeffs));
  for (PetscInt q = 0; q < twos_p1; ++q) r_coeffs[q] += c0_v[q];
  for (PetscInt q = 0; q < twos_p2; ++q) r_coeffs[q] -= tau1 * c1_v[q];
  for (PetscInt q = 0; q < mp1;     ++q) r_coeffs[q] -= tau2 * c2_v[q];

  PetscCall(VecSet(r0, 0.0));
  for (PetscInt k = 0; k < mp1; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
    PetscCall(VecAXPY(r0, r_coeffs[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
  }

  /* ---- chk16: FINAL state ------------------------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private  (16, "V0_final",   gms->V0));
  PetscCall(KSPGMSTABDumpMat_Private  (16, "V1_final",   gms->V1));
  PetscCall(KSPGMSTABDumpDense_Private(16, "Z_final",    gms->Z, gms->Z_ldim, s, s));
  PetscCall(KSPGMSTABDumpVec_Private  (16, "r0_final",   r0));
  PetscCall(KSPGMSTABDumpReal_Private (16, "beta_final", beta));

  /* No final snapshot inside cycle2 — the C++ port's gmstab2 has only
     ONE perf.read (the post-x-update one at solver.cpp:436-438, which
     we emitted at chk07 above). The driver's post-cycle snapshot
     follows the function return. This is asymmetric with gmstab1
     (which has a final perf.read at solver.cpp:239) and matches the
     C++ port's gmstab2 structure exactly. */

  PetscCall(PetscFree(r_coeffs));
  PetscCall(PetscFree(QfQz));
  PetscCall(PetscFree(coeffs));
  PetscCall(PetscFree(Qg_Rf_Qz));
  PetscCall(PetscFree(Rf_solve_Qz));
  PetscCall(PetscFree(big_combined));
  PetscCall(PetscFree(big_t2));
  PetscCall(PetscFree(big_t1));
  PetscCall(PetscFree(big_top));
  PetscCall(PetscFree2(Qz, Lz_new));
  PetscCall(PetscFree(tmp_lq));
  PetscCall(PetscFree(H1_H0_QgRf));
  PetscCall(PetscFree(H0_QgRf));
  PetscCall(PetscFree(QgRf));
  PetscCall(PetscFree2(Qf, Rf));
  PetscCall(PetscFree(F));
  PetscCall(PetscFree(RwC));
  PetscCall(PetscFree(Zsolve_YH1H0));
  PetscCall(PetscFree(Y_H1H0));
  PetscCall(PetscFree(H2_H1_H0));
  PetscCall(PetscFree(H1_H0));
  PetscCall(PetscFree(C_blk));
  PetscCall(PetscFree(term3));
  PetscCall(PetscFree(term2));
  PetscCall(PetscFree(term1));
  PetscCall(PetscFree(Qg));
  PetscCall(PetscFree(H0t_Qyt));
  PetscCall(PetscFree(Rw));
  PetscCall(PetscFree(Zinv_Yxi));
  PetscCall(PetscFree(Yxi));
  PetscCall(PetscFree(lin_term));
  PetscCall(PetscFree(c0_pad));
  PetscCall(PetscFree(xi_pad));
  PetscCall(PetscFree(c2_v));
  PetscCall(PetscFree(c1_v));
  PetscCall(PetscFree(c0_v));
  PetscCall(PetscFree(xi));
  PetscCall(PetscFree(inner));
  PetscCall(PetscFree(H1gamma));
  PetscCall(PetscFree(H1Qh0));
  PetscCall(PetscFree(bot_blk));
  PetscCall(PetscFree(top_blk));
  PetscCall(PetscFree(rhs_stack));
  PetscCall(PetscFree(M_stack));
  PetscCall(PetscFree(gamma_vec));
  PetscCall(PetscFree(H2m));
  PetscCall(PetscFree(H1m));
  PetscCall(PetscFree(Rh0));
  PetscCall(PetscFree(Qh0_));
  PetscCall(PetscFree(H0));
  PetscCall(PetscFree(Qy));
  PetscCall(PetscFree(Qyh));
  PetscCall(PetscFree(Qbuf));
  PetscCall(PetscFree(Yt));

  *beta_io = beta;
  PetscFunctionReturn(PETSC_SUCCESS);
}
