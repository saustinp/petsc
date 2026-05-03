/*
   GMstab1 — L=1 cycle. Direct port of gmstab_cpp/src/solver.cpp::gmstab1
   (lines 109-241) which itself is a near line-by-line port of GMstab1.m.

   Algorithm summary (single cycle):

     1. Compute r1 = A * r0, then [tau, beta_new] = StabCoeffs([r0, r1], beta).
        L=1 → tau is a 1-vector.
        x   ← x + tau * r0
        r0  ← r0 - tau * r1
        beta := beta_new
     2. If beta < tolabs: snapshot, return.
     3. Block-Gram-Schmidt extension of V0 against [V1] producing C ((2s)x(2s)):
          C(0..s, 0..s) = I_s
          For i=0..s-1:
            For j=0..s-1: C(j, s+i) = V1(:, j)' V0(:, i); V0(:, i) -= C V1(:, j)
            For j=0..i-1: C(s+j, s+i) = V0(:, j)' V0(:, i); V0(:, i) -= C V0(:, j)
            C(s+i, s+i) = ||V0(:, i)||;  V0(:, i) /= C(s+i, s+i)
     4. F = (1/tau) * C(:, s..2s-1) - C(:, 0..s-1)            (2s × s)
        [Q_f, R_f] = qr(F) → Q_f is 2s×s thin Q, R_f is s×s upper-tri.
     5. Z update:
          ZbyRf = -Z / R_f                  (Z * inv(R_f), implemented as
                                              (R_f' \ -Z') -> transpose)
          [Q_z, L_z] = lq(ZbyRf)
          Z := L_z                           (update gms->Z)
     6. V0 := [V1, V0] * (Q_f * Q_z)         (N × s)
     7. (W, H, Q_h, R_h, Y) := pgmres_m(A, P, Z, V0, x, r0, beta, m=s)
                                                (s counted matvecs)
        — and snapshot per inner-iteration via the snap callback.
     8. If beta < tolabs OR perf.isTerminate(): return.
     9. eta := beta * Y(:, 0)
        gamma := [beta; 0; ...; 0]   (s+1 × 1)
        [Q_z, L_z] := lq(Y * Q_h)
        xi := R_h \ (Q_z * (L_z \ eta))
     10. x   += W(:, 0..s-1) * xi - V0 * (Z \ (Y(:, 0..s-1) * xi))
         c0   := gamma - H * xi             (s+1 × 1)
         beta := ||c0||
         V0   := W(:, 0..s-1) * (R_h \ Q_z) - V0 * (Z \ (Y(:, 0..s-1) * (R_h \ Q_z)))
         V1   := W(:, 0..s) * (Q_h * Q_z)
         r0   := W(:, 0..s) * c0
         Z    := L_z
     11. Snapshot.
*/
#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_internal.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_dump.h>
#include <math.h>

/* Snapshot callback used by pgmres to fire mid-iteration snapshots. */
typedef struct {
  KSP         ksp;
  KSP_GMSTAB *gms;
  Vec         x_outer;   /* the x_local at cycle start (passed unchanged to
                            pgmres_m which reconstructs x_tmp = x_outer + ...) */
} cycle1_snap_ctx_t;

static PetscErrorCode cycle1_snap_cb(KSP ksp, Vec x_global_plus_local, PetscReal iter_norm,
                                      PetscBool *should_terminate, void *ctx_v)
{
  PetscFunctionBegin;
  cycle1_snap_ctx_t *ctx = (cycle1_snap_ctx_t *)ctx_v;
  PetscCall(KSPGMSTABSnapshot_Private(ctx->ksp, ctx->gms, x_global_plus_local, iter_norm));
  *should_terminate = (PetscBool)(ksp->reason != KSP_CONVERGED_ITERATING);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPGMSTABCycle1_Private(KSP ksp, KSP_GMSTAB *gms,
                                                     KSPGMSTABInnerWorkspace *ws,
                                                     Vec x_local, Vec r0, PetscReal *beta_io)
{
  PetscFunctionBegin;
  const PetscInt s = gms->s;
  PetscReal      beta = *beta_io;

  /* ---- chk00: ENTRY ----------------------------------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private (0, "V0_in",   gms->V0));
  PetscCall(KSPGMSTABDumpMat_Private (0, "V1_in",   gms->V1));
  PetscCall(KSPGMSTABDumpDense_Private(0, "Z_in",   gms->Z, gms->Z_ldim, s, s));
  PetscCall(KSPGMSTABDumpVec_Private (0, "x_in",   x_local));
  PetscCall(KSPGMSTABDumpVec_Private (0, "r0_in",  r0));
  PetscCall(KSPGMSTABDumpReal_Private(0, "beta_in", beta));

  /* (1) r1 = A * r0  [counted matvec], StabCoeffs on [r0, r1]. */
  Vec r1;
  PetscCall(VecDuplicate(r0, &r1));
  PetscCall(KSP_PCApplyBAorAB(ksp, r0, r1, ws->work_n));
  gms->matvec_count++;

  /* ---- chk01: AFTER r1 = A * r0 ---------------------------------------*/
  PetscCall(KSPGMSTABDumpVec_Private(1, "r1", r1));

  Vec stab_cols[2] = {r0, r1};
  PetscScalar tau_buf[1];
  PetscReal beta_after_stab;
  PetscCall(KSPGMSTABStabCoeffs_Private(stab_cols, /*L=*/1, beta, gms->stab_angle, tau_buf, &beta_after_stab));
  const PetscScalar tau = tau_buf[0];
  beta = beta_after_stab;

  /* ---- chk02: AFTER StabCoeffs ----------------------------------------*/
  PetscCall(KSPGMSTABDumpReal_Private(2, "tau",      (PetscReal)PetscRealPart(tau)));
  PetscCall(KSPGMSTABDumpReal_Private(2, "beta_new", beta));

  /* x ← x + tau*r0 ; r0 ← r0 - tau*r1 */
  PetscCall(VecAXPY(x_local, tau, r0));
  PetscCall(VecAXPY(r0, -tau, r1));
  PetscCall(VecDestroy(&r1));

  /* ---- chk03: AFTER applying tau --------------------------------------*/
  PetscCall(KSPGMSTABDumpVec_Private(3, "x",  x_local));
  PetscCall(KSPGMSTABDumpVec_Private(3, "r0", r0));

  /* C++ solver.cpp:171-174 returns from cycle1 here WITHOUT emitting a
     perf.read when beta has already crossed below tolabs (the polynomial
     step at chk03 happens to bring beta into convergence). The driver
     then breaks at "if (beta <= tolabs)". Mirror that exactly: skip the
     mid-cycle snapshot in this branch, so the trace stays bit-equivalent
     under any path that exits the cycle here. */
  if (beta < ksp->abstol) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* Mid-cycle snapshot per the C++ port (solver.cpp:175-177). */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
    *beta_io = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* (3) Block-Gram-Schmidt extension of V0 against [V1].
         C is 2s × 2s, lower-tri-blocks and identity layout per the
         C++ port (line 142-156).

         Note on rank deficiency: V0 and V1 are both s-dim subspaces of
         the same (s+1)-dim Krylov space (built by Initialisation), so
         range(V0) ∩ range(V1) is at least (s−1)-dim. The BGS will
         find that V0 columns 1..s-1 reduce to magnitude ~1e-16 after
         orthogonalization (only column 0 captures the one fresh
         direction). The MATLAB reference (GMstab1.m lines 14-29) and
         the C++ port both unconditionally normalize each column
         regardless of its post-orthogonalization norm — there is NO
         rank-deficiency detection or skip path. We do the same. The
         resulting noise-direction unit vectors are then absorbed by
         the QfQz multiplication at step (6): the rows of QfQz that
         pair with the noise V0 columns inherit machine-epsilon
         magnitude from F's rank-deficient columns, so noise × ε
         vanishes against the legitimate V1 contributions. See
         PHASE3_STATUS.md "V0_postBGS rank-deficiency noise" for the
         full lifecycle. */
  PetscScalar *C;
  const PetscInt twos = 2 * s;
  PetscCall(PetscCalloc1((size_t)twos * (size_t)twos, &C));
  /* C(0..s-1, 0..s-1) = I_s. */
  for (PetscInt i = 0; i < s; ++i) C[(size_t)i + (size_t)i * (size_t)twos] = 1.0;

  for (PetscInt i = 0; i < s; ++i) {
    /* Get V0 column i (we'll mutate it). */
    Vec V0i;
    PetscCall(MatDenseGetColumnVec(gms->V0, i, &V0i));
    /* Orthogonalise against V1 columns 0..s-1. */
    for (PetscInt j = 0; j < s; ++j) {
      Vec V1j;
      PetscScalar c_ji;
      PetscCall(MatDenseGetColumnVecRead(gms->V1, j, &V1j));
      PetscCall(VecDot(V0i, V1j, &c_ji));
      PetscCall(VecAXPY(V0i, -c_ji, V1j));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V1, j, &V1j));
      C[(size_t)j + (size_t)(s + i) * (size_t)twos] = c_ji;
    }
    /* Orthogonalise against earlier V0 columns (j < i). */
    PetscCall(MatDenseRestoreColumnVec(gms->V0, i, &V0i));     /* release before re-borrow elsewhere */
    /* For each j < i: read V0[:,j], modify V0[:,i] — same matrix double-borrow.
       Workaround: copy V0[:,i] into work_out, do GS on work_out, write back. */
    PetscCall(MatDenseGetColumnVecRead(gms->V0, i, &V0i));
    PetscCall(VecCopy(V0i, ws->work_out));
    PetscCall(MatDenseRestoreColumnVecRead(gms->V0, i, &V0i));

    for (PetscInt j = 0; j < i; ++j) {
      Vec V0j;
      PetscScalar c_sji;
      PetscCall(MatDenseGetColumnVecRead(gms->V0, j, &V0j));
      PetscCall(VecDot(ws->work_out, V0j, &c_sji));
      PetscCall(VecAXPY(ws->work_out, -c_sji, V0j));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V0, j, &V0j));
      C[(size_t)(s + j) + (size_t)(s + i) * (size_t)twos] = c_sji;
    }
    PetscReal nrm;
    PetscCall(VecNorm(ws->work_out, NORM_2, &nrm));
    C[(size_t)(s + i) + (size_t)(s + i) * (size_t)twos] = (PetscScalar)nrm;
    PetscCall(VecScale(ws->work_out, 1.0 / nrm));
    /* Write back into V0[:, i]. */
    PetscCall(MatDenseGetColumnVec(gms->V0, i, &V0i));
    PetscCall(VecCopy(ws->work_out, V0i));
    PetscCall(MatDenseRestoreColumnVec(gms->V0, i, &V0i));
  }

  /* ---- chk04: AFTER block-GS extension --------------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private  (4, "V0_postBGS", gms->V0));
  PetscCall(KSPGMSTABDumpDense_Private(4, "C", C, twos, twos, twos));

  /* (4) F = (1/tau) * C(:, s..2s-1) - C(:, 0..s-1)   (2s × s)
         [Q_f, R_f] = qr(F) — Q_f is 2s × s thin, R_f is s × s upper-tri. */
  PetscScalar *F;
  PetscCall(PetscMalloc1((size_t)twos * (size_t)s, &F));
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < twos; ++rr)
      F[(size_t)rr + (size_t)c_ * (size_t)twos] =
          (1.0 / tau) * C[(size_t)rr + (size_t)(s + c_) * (size_t)twos]
        -              C[(size_t)rr + (size_t)c_ * (size_t)twos];

  /* ---- chk05: AFTER F construction ------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(5, "F", F, twos, twos, s));

  /* QR(F): use dgeqrf+dorgqr to extract thin Q (2s × s) and R (s × s). */
  PetscScalar *Qf, *Rf;
  PetscCall(PetscCalloc2((size_t)twos * (size_t)s, &Qf, (size_t)s * (size_t)s, &Rf));
  {
    PetscScalar *Fbuf;
    PetscCall(PetscMalloc1((size_t)twos * (size_t)s, &Fbuf));
    for (PetscInt q = 0; q < twos * s; ++q) Fbuf[q] = F[q];

    PetscBLASInt m_, n_, lda_ = (PetscBLASInt)twos, lwork, info, k_ = (PetscBLASInt)s;
    PetscCall(PetscBLASIntCast(twos, &m_));
    PetscCall(PetscBLASIntCast(s,    &n_));

    PetscScalar *tau_v, wkq;
    PetscCall(PetscMalloc1(s, &tau_v));
    lwork = -1;
    PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Fbuf, &lda_, tau_v, &wkq, &lwork, &info));
    lwork = (PetscBLASInt)PetscRealPart(wkq);
    PetscScalar *work;
    PetscCall(PetscMalloc1(lwork, &work));
    PetscCallBLAS("LAPACKgeqrf", LAPACKgeqrf_(&m_, &n_, Fbuf, &lda_, tau_v, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "geqrf F: info=%d", (int)info);
    PetscCall(PetscFree(work));

    /* Extract R (upper triangle of top s×s of Fbuf). */
    for (PetscInt c_ = 0; c_ < s; ++c_)
      for (PetscInt rr = 0; rr <= c_; ++rr)
        Rf[(size_t)rr + (size_t)c_ * (size_t)s] = Fbuf[(size_t)rr + (size_t)c_ * (size_t)twos];

    /* Reify thin Q: dorgqr expects (m, n=k, k) with output 2s × s. */
    lwork = -1;
    PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &n_, &k_, Fbuf, &lda_, tau_v, &wkq, &lwork, &info));
    lwork = (PetscBLASInt)PetscRealPart(wkq);
    PetscCall(PetscMalloc1(lwork, &work));
    PetscCallBLAS("LAPACKorgqr", LAPACKorgqr_(&m_, &n_, &k_, Fbuf, &lda_, tau_v, work, &lwork, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "orgqr F: info=%d", (int)info);
    PetscCall(PetscFree(work));

    /* Copy Fbuf (2s × s) into Qf. */
    for (PetscInt q = 0; q < twos * s; ++q) Qf[q] = Fbuf[q];

    PetscCall(PetscFree(tau_v));
    PetscCall(PetscFree(Fbuf));
  }

  /* ---- chk06: AFTER QR(F) ---------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(6, "Qf", Qf, twos, twos, s));
  PetscCall(KSPGMSTABDumpDense_Private(6, "Rf", Rf, s,    s,    s));

  /* (5) ZbyRf := -Z / R_f.   We solve X * R_f = -Z  →  X = -Z * inv(R_f).
         Implemented as: solve R_f' * X' = -Z' (lower-tri), then transpose. */
  PetscScalar *ZbyRf;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &ZbyRf));
  /* Build -Z' into ZbyRf_t (s × s, layout used as RHS for the lower-tri
     solve). Then dtrsm with side='L', uplo='L', trans='T' actually solves
     (R_f' transposed, i.e. R_f) ... too confusing. Use the alternative:
       Step a: copy -Z into a buffer ZB (still s × s)
       Step b: trsm(side='R', uplo='U', trans='N') solves X * R_f = -Z in place.
     That's the standard way. */
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < s; ++rr)
      ZbyRf[(size_t)rr + (size_t)c_ * (size_t)s] = -gms->Z[(size_t)rr + (size_t)c_ * (size_t)s];
  PetscCall(KSPGMSTABTrsm_Private("R", "U", "N", "N", s, s, 1.0, Rf, s, ZbyRf, s));

  /* ---- chk07: AFTER ZbyRf trsm ----------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(7, "ZbyRf", ZbyRf, s, s, s));

  /* [Q_z, L_z] := lq(ZbyRf). */
  PetscScalar *Qz, *Lz;
  PetscCall(PetscMalloc2((size_t)s * (size_t)s, &Qz, (size_t)s * (size_t)s, &Lz));
  PetscCall(KSPGMSTABLq_Private(ZbyRf, s, s, s, Qz, s, Lz, s));

  /* Update gms->Z := L_z. */
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < s; ++rr)
      gms->Z[(size_t)rr + (size_t)c_ * (size_t)s] = Lz[(size_t)rr + (size_t)c_ * (size_t)s];

  /* ---- chk08: AFTER lq(ZbyRf) -----------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(8, "Qz",       Qz,    s, s, s));
  PetscCall(KSPGMSTABDumpDense_Private(8, "Z_postLq", Lz,    s, s, s));

  /* (6) V0 := [V1, V0] * (Q_f * Q_z).   QfQz is 2s × s. */
  PetscScalar *QfQz;
  PetscCall(PetscMalloc1((size_t)twos * (size_t)s, &QfQz));
  PetscCall(KSPGMSTABGemm_Private("N", "N", twos, s, s, 1.0, Qf, twos, Qz, s, 0.0, QfQz, twos));

  /* ---- chk09: AFTER Qf * Qz product -----------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(9, "QfQz", QfQz, twos, twos, s));

  /* Build a temporary V_concat = [V1, V0] (N × 2s) by using two scratch
     vec accumulations per output column. We produce the new V0 column-
     by-column into a scratch then write back. */
  Mat V0_new;
  {
    PetscInt n_local;
    PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &V0_new));
    PetscCall(MatSetType(V0_new, MATDENSE));
    PetscCall(MatSetSizes(V0_new, n_local, PETSC_DECIDE, N_global, s));
    PetscCall(MatSetUp(V0_new));
  }
  for (PetscInt c_ = 0; c_ < s; ++c_) {
    Vec V0nc;
    PetscCall(MatDenseGetColumnVec(V0_new, c_, &V0nc));
    PetscCall(VecSet(V0nc, 0.0));
    /* First s "rows" of QfQz are coefficients on V1 columns. */
    for (PetscInt k = 0; k < s; ++k) {
      Vec V1k;
      PetscCall(MatDenseGetColumnVecRead(gms->V1, k, &V1k));
      PetscCall(VecAXPY(V0nc, QfQz[(size_t)k + (size_t)c_ * (size_t)twos], V1k));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V1, k, &V1k));
    }
    /* Last s "rows" of QfQz are coefficients on V0 columns. */
    for (PetscInt k = 0; k < s; ++k) {
      Vec V0k;
      PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
      PetscCall(VecAXPY(V0nc, QfQz[(size_t)(s + k) + (size_t)c_ * (size_t)twos], V0k));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
    }
    PetscCall(MatDenseRestoreColumnVec(V0_new, c_, &V0nc));
  }
  /* Replace V0 with V0_new. */
  PetscCall(MatDestroy(&gms->V0));
  gms->V0 = V0_new;

  /* ---- chk10: AFTER V0 := [V1, V0] * QfQz -----------------------------*/
  PetscCall(KSPGMSTABDumpMat_Private(10, "V0_postUpdate", gms->V0));

  /* (7) pgmres_m for s steps. Reset workspace first. */
  ws->m = s;
  {
    const PetscInt mp1 = s + 1;
    PetscCall(PetscArrayzero(ws->H,     (size_t)mp1 * (size_t)s));
    PetscCall(PetscArrayzero(ws->R,     (size_t)mp1 * (size_t)s));
    PetscCall(PetscArrayzero(ws->Q,     (size_t)mp1 * (size_t)mp1));
    PetscCall(PetscArrayzero(ws->Y,     (size_t)s   * (size_t)mp1));
    PetscCall(PetscArrayzero(ws->gamma, (size_t)mp1));
    PetscCall(PetscArrayzero(ws->G,     (size_t)s));
    for (PetscInt q = 0; q < mp1; ++q) ws->Q[(size_t)q + (size_t)q * (size_t)mp1] = 1.0;
  }

  cycle1_snap_ctx_t snap_ctx = {ksp, gms, x_local};
  Vec x_p, r_p;
  PetscCall(VecDuplicate(x_local, &x_p));
  PetscCall(VecDuplicate(r0,      &r_p));
  PetscBool inner_converged = PETSC_FALSE, inner_terminated = PETSC_FALSE;
  PetscReal beta_after_p;
  PetscCall(KSPGMSTABPGmresM_Private(ksp, gms->P, gms->Z, gms->Z_ldim,
            gms->V0,
            x_local, r0, beta,
            ksp->abstol, ws,
            gms->x_global, cycle1_snap_cb, &snap_ctx,
            x_p, r_p, &beta_after_p, &inner_converged, &inner_terminated));

  /* Adopt the inner-solver outputs. */
  PetscCall(VecCopy(x_p, x_local));
  PetscCall(VecCopy(r_p, r0));
  beta = beta_after_p;
  PetscCall(VecDestroy(&x_p));
  PetscCall(VecDestroy(&r_p));

  /* ---- chk11: AFTER pgmres_m ------------------------------------------
     We dump the workspace state AFTER pgmres returns: ws->W, ws->Y, ws->H,
     plus the extracted Qh and Rh that the cycle uses below. Note Qh in
     the C++ port is (s+1) × s = Q.topRows(m).transpose(); we extract from
     ws->Q below (line by line in the post-update step), but for the dump
     we materialize the same form here. */
  {
    const PetscInt mp1_dump = s + 1;
    /* Y, H are stored in ws->Y / ws->H as column-major s × mp1, mp1 × s
       respectively (with leading dim ldH = s for Y, mp1 for H). */
    /* ws->W is allocated with m_max+1 = 2s+3 columns; cycle1 only fills
       the first s+1. Dump just the active slice so it pairs cleanly with
       the C++ port's (s+1)-column W. ws->W is a *distributed* MatDense,
       so we must use the parallel-aware Mat dump, not the replicated
       small-dense path (which would silently emit only rank 0's local
       rows in parallel mode). */
    PetscCall(KSPGMSTABDumpMatCols_Private(11, "W", ws->W, mp1_dump));
    PetscCall(KSPGMSTABDumpDense_Private(11, "Y", ws->Y, s,        s,    mp1_dump));
    PetscCall(KSPGMSTABDumpDense_Private(11, "H", ws->H, mp1_dump, mp1_dump, s));
    /* Extract Qh same way as the post-update logic does, so we dump
       *exactly* what the cycle code uses. */
    PetscScalar *Qh_dump;
    PetscCall(PetscMalloc1((size_t)mp1_dump * (size_t)s, &Qh_dump));
    for (PetscInt b_ = 0; b_ < s; ++b_)
      for (PetscInt a_ = 0; a_ < mp1_dump; ++a_)
        Qh_dump[(size_t)a_ + (size_t)b_ * (size_t)mp1_dump] =
            ws->Q[(size_t)b_ + (size_t)a_ * (size_t)mp1_dump];
    PetscCall(KSPGMSTABDumpDense_Private(11, "Qh", Qh_dump, mp1_dump, mp1_dump, s));
    PetscCall(PetscFree(Qh_dump));
    /* Rh is the upper triangle of ws->R(0:s, 0:s) (column-major, ldR = mp1). */
    PetscScalar *Rh_dump;
    PetscCall(PetscCalloc1((size_t)s * (size_t)s, &Rh_dump));
    for (PetscInt b_ = 0; b_ < s; ++b_)
      for (PetscInt a_ = 0; a_ <= b_; ++a_)
        Rh_dump[(size_t)a_ + (size_t)b_ * (size_t)s] =
            ws->R[(size_t)a_ + (size_t)b_ * (size_t)mp1_dump];
    PetscCall(KSPGMSTABDumpDense_Private(11, "Rh", Rh_dump, s, s, s));
    PetscCall(PetscFree(Rh_dump));
    PetscCall(KSPGMSTABDumpVec_Private (11, "x_postP", x_local));
    PetscCall(KSPGMSTABDumpVec_Private (11, "r_postP", r0));
    PetscCall(KSPGMSTABDumpReal_Private(11, "beta_postP", beta));
  }

  /* (8) Convergence check / termination. */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));
  if (beta < ksp->abstol || (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) || inner_terminated) {
    *beta_io = beta;
    PetscCall(PetscFree(QfQz));
    PetscCall(PetscFree2(Qz, Lz));
    PetscCall(PetscFree(ZbyRf));
    PetscCall(PetscFree2(Qf, Rf));
    PetscCall(PetscFree(F));
    PetscCall(PetscFree(C));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* (9) Post-pgmres update — the second half of GMstab1.m.
         eta = beta * Y(:, 0)
         gamma_vec = [beta; 0; ...; 0]        (s+1 × 1)
         [Q_z2, L_z2] = lq(Y * Q_h)            but Q_h here is the inner-
           GMRES Q output (which we extract from ws->Q.topRows(s).T form). */
  const PetscInt mp1 = s + 1;
  PetscScalar *eta;
  PetscCall(PetscMalloc1(s, &eta));
  for (PetscInt k = 0; k < s; ++k) eta[k] = (PetscScalar)beta * ws->Y[(size_t)k];

  PetscScalar *Qh;
  PetscCall(PetscMalloc1((size_t)mp1 * (size_t)s, &Qh));
  for (PetscInt b_ = 0; b_ < s; ++b_)
    for (PetscInt a_ = 0; a_ < mp1; ++a_)
      Qh[(size_t)a_ + (size_t)b_ * (size_t)mp1] = ws->Q[(size_t)b_ + (size_t)a_ * (size_t)mp1];

  PetscScalar *Rh;
  PetscCall(PetscCalloc1((size_t)s * (size_t)s, &Rh));
  for (PetscInt b_ = 0; b_ < s; ++b_)
    for (PetscInt a_ = 0; a_ <= b_; ++a_)
      Rh[(size_t)a_ + (size_t)b_ * (size_t)s] = ws->R[(size_t)a_ + (size_t)b_ * (size_t)mp1];

  /* Y * Q_h is s × s. */
  PetscScalar *YQh;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &YQh));
  PetscCall(KSPGMSTABGemm_Private("N", "N", s, s, mp1, 1.0, ws->Y, s, Qh, mp1, 0.0, YQh, s));

  /* ---- chk12: Y * Qh --------------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(12, "YQh", YQh, s, s, s));

  PetscScalar *Qz2, *Lz2;
  PetscCall(PetscMalloc2((size_t)s * (size_t)s, &Qz2, (size_t)s * (size_t)s, &Lz2));
  PetscCall(KSPGMSTABLq_Private(YQh, s, s, s, Qz2, s, Lz2, s));

  /* ---- chk13: AFTER lq(Y*Qh) ------------------------------------------*/
  PetscCall(KSPGMSTABDumpDense_Private(13, "Qz2", Qz2, s, s, s));
  PetscCall(KSPGMSTABDumpDense_Private(13, "Lz2", Lz2, s, s, s));

  /* xi = Rh \ (Qz2 * (Lz2 \ eta)). */
  PetscScalar *inner, *xi, *temp;
  PetscCall(PetscMalloc3(s, &inner, s, &xi, s, &temp));
  for (PetscInt k = 0; k < s; ++k) inner[k] = eta[k];
  PetscCall(KSPGMSTABTrsv_Private("L", "N", "N", s, Lz2, s, inner, 1));
  PetscCall(KSPGMSTABGemv_Private("N", s, s, 1.0, Qz2, s, inner, 1, 0.0, temp, 1));
  for (PetscInt k = 0; k < s; ++k) xi[k] = temp[k];
  PetscCall(KSPGMSTABTrsv_Private("U", "N", "N", s, Rh, s, xi, 1));

  /* ---- chk14: AFTER xi ------------------------------------------------*/
  PetscCall(KSPGMSTABDumpVector1d_Private(14, "eta",   eta,   s));
  PetscCall(KSPGMSTABDumpVector1d_Private(14, "inner", inner, s));
  PetscCall(KSPGMSTABDumpVector1d_Private(14, "xi",    xi,    s));

  /* (10) x += W(:, 0..s-1) * xi - V0 * (Z \ (Y(:, 0..s-1) * xi)). */
  for (PetscInt k = 0; k < s; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
    PetscCall(VecAXPY(x_local, xi[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
  }
  /* Z \ (Y(:, 0..s-1) * xi) — Y here is the full s × (s+1), use first s cols. */
  PetscScalar *Yxi, *ZinvYxi;
  PetscCall(PetscMalloc2(s, &Yxi, s, &ZinvYxi));
  PetscCall(KSPGMSTABGemv_Private("N", s, s, 1.0, ws->Y, s, xi, 1, 0.0, Yxi, 1));
  for (PetscInt k = 0; k < s; ++k) ZinvYxi[k] = Yxi[k];
  PetscCall(KSPGMSTABTrsv_Private("L", "N", "N", s, gms->Z, gms->Z_ldim, ZinvYxi, 1));
  for (PetscInt k = 0; k < s; ++k) {
    Vec V0k;
    PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
    PetscCall(VecAXPY(x_local, -ZinvYxi[k], V0k));
    PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
  }

  /* ---- chk15: AFTER x update ------------------------------------------*/
  PetscCall(KSPGMSTABDumpVec_Private(15, "x", x_local));

  /* c0 = gamma_vec - H * xi;    beta = ||c0||. */
  PetscScalar *c0;
  PetscCall(PetscMalloc1(mp1, &c0));
  c0[0] = (PetscScalar)beta;
  for (PetscInt k = 1; k < mp1; ++k) c0[k] = 0.0;
  PetscCall(KSPGMSTABGemv_Private("N", mp1, s, -1.0, ws->H, mp1, xi, 1, 1.0, c0, 1));
  {
    PetscReal cn = 0.0;
    for (PetscInt k = 0; k < mp1; ++k) cn += PetscRealPart(c0[k]) * PetscRealPart(c0[k]);
    beta = PetscSqrtReal(cn);
  }

  /* ---- chk16: AFTER c0 / beta -----------------------------------------*/
  PetscCall(KSPGMSTABDumpVector1d_Private(16, "c0", c0, mp1));
  PetscCall(KSPGMSTABDumpReal_Private    (16, "beta_final", beta));

  /* V0 := W(:, 0..s-1) * (Rh \ Qz2) - V0 * (Z \ (Y(:, 0..s-1) * (Rh \ Qz2))). */
  PetscScalar *RhQz2;
  PetscCall(PetscMalloc1((size_t)s * (size_t)s, &RhQz2));
  for (PetscInt q = 0; q < s * s; ++q) RhQz2[q] = Qz2[q];
  PetscCall(KSPGMSTABTrsm_Private("L", "U", "N", "N", s, s, 1.0, Rh, s, RhQz2, s));

  /* YRhQz = Y(:, 0..s-1) * RhQz2  (s × s). */
  PetscScalar *YRhQz, *ZinvYRhQz;
  PetscCall(PetscMalloc2((size_t)s * (size_t)s, &YRhQz, (size_t)s * (size_t)s, &ZinvYRhQz));
  PetscCall(KSPGMSTABGemm_Private("N", "N", s, s, s, 1.0, ws->Y, s, RhQz2, s, 0.0, YRhQz, s));
  for (PetscInt q = 0; q < s * s; ++q) ZinvYRhQz[q] = YRhQz[q];
  PetscCall(KSPGMSTABTrsm_Private("L", "L", "N", "N", s, s, 1.0, gms->Z, gms->Z_ldim, ZinvYRhQz, s));

  /* New V0_new[:, c] = sum_k W[:, k] * RhQz2[k, c] - sum_k V0[:, k] * ZinvYRhQz[k, c]. */
  Mat V0_post;
  {
    PetscInt n_local; PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &V0_post));
    PetscCall(MatSetType(V0_post, MATDENSE));
    PetscCall(MatSetSizes(V0_post, n_local, PETSC_DECIDE, N_global, s));
    PetscCall(MatSetUp(V0_post));
  }
  for (PetscInt c_ = 0; c_ < s; ++c_) {
    Vec V0pc;
    PetscCall(MatDenseGetColumnVec(V0_post, c_, &V0pc));
    PetscCall(VecSet(V0pc, 0.0));
    for (PetscInt k = 0; k < s; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V0pc, RhQz2[(size_t)k + (size_t)c_ * (size_t)s], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    for (PetscInt k = 0; k < s; ++k) {
      Vec V0k;
      PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
      PetscCall(VecAXPY(V0pc, -ZinvYRhQz[(size_t)k + (size_t)c_ * (size_t)s], V0k));
      PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
    }
    PetscCall(MatDenseRestoreColumnVec(V0_post, c_, &V0pc));
  }
  PetscCall(MatDestroy(&gms->V0));
  gms->V0 = V0_post;

  /* V1 := W(:, 0..s) * (Q_h * Q_z2). */
  PetscScalar *QhQz2;
  PetscCall(PetscMalloc1((size_t)mp1 * (size_t)s, &QhQz2));
  PetscCall(KSPGMSTABGemm_Private("N", "N", mp1, s, s, 1.0, Qh, mp1, Qz2, s, 0.0, QhQz2, mp1));

  Mat V1_new;
  {
    PetscInt n_local; PetscInt N_global;
    PetscCall(VecGetLocalSize(x_local, &n_local));
    PetscCall(VecGetSize(x_local, &N_global));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)ksp), &V1_new));
    PetscCall(MatSetType(V1_new, MATDENSE));
    PetscCall(MatSetSizes(V1_new, n_local, PETSC_DECIDE, N_global, s));
    PetscCall(MatSetUp(V1_new));
  }
  for (PetscInt c_ = 0; c_ < s; ++c_) {
    Vec V1nc;
    PetscCall(MatDenseGetColumnVec(V1_new, c_, &V1nc));
    PetscCall(VecSet(V1nc, 0.0));
    for (PetscInt k = 0; k <= s; ++k) {
      Vec wk;
      PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
      PetscCall(VecAXPY(V1nc, QhQz2[(size_t)k + (size_t)c_ * (size_t)mp1], wk));
      PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
    }
    PetscCall(MatDenseRestoreColumnVec(V1_new, c_, &V1nc));
  }
  PetscCall(MatDestroy(&gms->V1));
  gms->V1 = V1_new;

  /* r0 = W(:, 0..s) * c0. */
  PetscCall(VecSet(r0, 0.0));
  for (PetscInt k = 0; k <= s; ++k) {
    Vec wk;
    PetscCall(MatDenseGetColumnVecRead(ws->W, k, &wk));
    PetscCall(VecAXPY(r0, c0[k], wk));
    PetscCall(MatDenseRestoreColumnVecRead(ws->W, k, &wk));
  }

  /* Z := L_z2. */
  for (PetscInt c_ = 0; c_ < s; ++c_)
    for (PetscInt rr = 0; rr < s; ++rr)
      gms->Z[(size_t)rr + (size_t)c_ * (size_t)s] = Lz2[(size_t)rr + (size_t)c_ * (size_t)s];

  /* ---- chk17: AFTER post-update (V0, V1, r0, Z) -----------------------*/
  PetscCall(KSPGMSTABDumpMat_Private  (17, "V0_final", gms->V0));
  PetscCall(KSPGMSTABDumpMat_Private  (17, "V1_final", gms->V1));
  PetscCall(KSPGMSTABDumpVec_Private  (17, "r0_final", r0));
  PetscCall(KSPGMSTABDumpDense_Private(17, "Z_final",  gms->Z, gms->Z_ldim, s, s));

  *beta_io = beta;

  /* (11) Final cycle snapshot. */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta));

  PetscCall(PetscFree(c0));
  PetscCall(PetscFree2(Yxi, ZinvYxi));
  PetscCall(PetscFree2(YRhQz, ZinvYRhQz));
  PetscCall(PetscFree(RhQz2));
  PetscCall(PetscFree3(inner, xi, temp));
  PetscCall(PetscFree2(Qz2, Lz2));
  PetscCall(PetscFree(YQh));
  PetscCall(PetscFree(Rh));
  PetscCall(PetscFree(Qh));
  PetscCall(PetscFree(eta));
  PetscCall(PetscFree(QhQz2));
  PetscCall(PetscFree(QfQz));
  PetscCall(PetscFree2(Qz, Lz));
  PetscCall(PetscFree(ZbyRf));
  PetscCall(PetscFree2(Qf, Rf));
  PetscCall(PetscFree(F));
  PetscCall(PetscFree(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}
