/*
   KSPGMSTAB — PETSc port of GM(s)stab. See gmstabimpl.h for the design
   summary, IMPLEMENTATION_PLAN.md for the phased plan, and PHASE0_VERIFICATION.md
   for the bit-equivalence oracle that drives Phase 5a verification.

   This file implements the public PETSc KSP type entry points
   (Create / SetUp / Solve / Destroy / Reset / SetFromOptions / View) plus
   the public option setters (KSPGMSTABSetS / KSPGMSTABSetShadowSpace /
   KSPGMSTABSetShadowSpaceFile).

   The KSPSolve_GMSTAB body itself is a phased implementation:

     Phase 1 (this commit): a stub that sets KSP_DIVERGED_BREAKDOWN with a
                            descriptive error and returns cleanly. This lets
                            the registration plumbing be exercised before
                            the algorithm is implemented.

     Phase 2-4:             progressively wire in pGMRESm / augGMRESm,
                            Initialisation, the L=1 / L=2 cycle bodies,
                            and preconditioner-side support.

     Phase 5a:              run all 129 baseline matrices against this
                            implementation and gate further work on
                            <= 1e-10 absolute drift vs the C++ port.
*/

#include <petsc/private/kspimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstabimpl.h>
#include <../src/ksp/ksp/impls/gmstab/gmstab_internal.h>
#include <petscblaslapack.h>
#include <math.h>

/* Default for std::mt19937_64 (the seed Eigen uses when seed=0 in the C++
   port's default_shadow_space). Defined here so it appears once. */
#define KSPGMSTAB_DEFAULT_SEED ((unsigned long)5489)
#define KSPGMSTAB_DEFAULT_STAB_ANGLE ((PetscReal)(M_PI * 7.0 / 36.0))

/* ============================================================================
 * KSPSolve_GMSTAB — driver. Port of Solver_GMsStab.m / solver.cpp::solve.
 *
 *   Mirrors the C++ port's outer driver:
 *
 *     1. Build / load shadow space P (one of three sources, validated in SetUp).
 *     2. Open trace CSV if requested; emit the initial (matvec=0, x=0, ||b||) row.
 *     3. Resume timer.
 *     4. b_local = b - A * xGlobal             [first counted matvec]
 *     5. r = b_local;  x_local = 0;  beta = ||r||;  betaLocal = beta.
 *     6. Initialisation(...).
 *     7. Snapshot.
 *     8. while beta > tolabs:
 *          decide t_restart, t_replace, L per the flying-restart heuristic
 *          if L == 1: gmstab1(...);    n2cycles = 0
 *          else:      gmstab2(...);    n2cycles += 1
 *          if (recycling enabled): hU = V0; recycling_index = snapshot_count
 *          if t_restart || t_replace: r = b_local - A*x; project; dir_rbio
 *          if t_restart: bLocal=r; xGlobal+=x; x=0; betaLocal=beta
 *          snapshot
 *          if KSPMonitor convergence -> break
 *
 * NOTE — Phase 3 status:
 *   Initialisation is implemented (gmstab_init.c).
 *   gmstab1 / gmstab2 cycle bodies are NOT yet implemented (Phase 3b).
 *   For the moment, if Initialisation does not converge (i.e. beta > tolabs
 *   after init), the driver returns KSP_DIVERGED_BREAKDOWN with a PetscInfo
 *   note. This lets us validate Initialisation in isolation while we
 *   continue with the cycle bodies in subsequent commits.
 * ============================================================================ */
static PetscErrorCode KSPSolve_GMSTAB(KSP ksp)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;
  PetscFunctionBegin;

  /* Reset per-solve counters and state. */
  gms->snapshot_count             = 0;
  gms->matvec_count               = 0;
  gms->_last_logged_matvec_count  = 0;
  gms->t_total                    = 0.0;
  gms->t_mv                       = 0.0;
  gms->cycle_count                = 0;
  gms->n2cycles                   = 0;
  ksp->its                        = 0;

  /* Shadow space: precedence  P_user > P_file > default RNG. */
  if (gms->P_user) {
    PetscCall(MatDestroy(&gms->P));
    PetscCall(PetscObjectReference((PetscObject)gms->P_user));
    gms->P = gms->P_user;
  } else if (gms->P_file[0]) {
    PetscCall(KSPGMSTABLoadShadowFile_Private(ksp, gms));
  } else {
    PetscCall(KSPGMSTABBuildDefaultShadow_Private(ksp, gms));
  }
  PetscCheck(gms->P, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB,
             "KSPGMSTAB: shadow space P was not built");

  /* Open trace CSV if requested. The C++ port writes the first snapshot
     in PerfMeasure's constructor (matvec=0, iterres=||b||, trueres=||b||,
     runtime=0); reproduce that exactly.

     Close-and-reopen on every KSPSolve so a second call to the same KSP
     object overwrites (rather than appends to) the trace file. This
     matches the bit-equivalence harness's expectation that one KSPSolve
     == one CSV. KSPReset_GMSTAB also closes the handle as a safety net. */
  if (gms->trace_csv) {
    if (gms->trace_fp) {
      fclose(gms->trace_fp);
      gms->trace_fp = NULL;
    }
    gms->trace_fp = fopen(gms->trace_csv_path, "w");
    PetscCheck(gms->trace_fp, PetscObjectComm((PetscObject)ksp), PETSC_ERR_FILE_OPEN,
               "Cannot open trace_csv: %s", gms->trace_csv_path);
    fprintf(gms->trace_fp, "iter,matvec,iterres,trueres,runtime,runtime_mv\n");
  }

  /* Set up the initial state.
     C++ does:
       perf init (writes initial snapshot at x=0, ||b||)
       resume()
       bLocal = b - A * xGlobal      [counted]
       r = bLocal;  x_local = 0; beta = ||r||
  */
  Vec b, x_local;
  PetscCall(KSPGetRhs(ksp, &b));
  x_local = ksp->vec_sol;

  /* Destroy any vectors / dense-mats / host buffers lingering from a
     previous KSPSolve before re-allocating. VecDuplicate / MatCreate into
     a non-NULL pointer would leak the prior object; this defends against
     multi-solve without an explicit KSPReset between. The Mats and Z buffer
     are also sized by `s`, so destroying them is the only safe way to
     handle a user that changes s with KSPGMSTABSetS between solves —
     Initialisation re-allocates V0/V1/Z lazily on the first cycle.

     gms->x_initial_guess (Phase 4a) is allocated/copied AFTER this destroy
     block on a per-solve basis when pc_side ∈ {PC_RIGHT, PC_SYMMETRIC};
     destroying here means a previous solve in PC_RIGHT mode can't leak
     stale state into a subsequent PC_NONE solve, and that the Vec is
     fresh-from-b on each invocation regardless of pc_side history. */
  PetscCall(VecDestroy(&gms->b_local));
  PetscCall(VecDestroy(&gms->x_global));
  PetscCall(VecDestroy(&gms->x_initial_guess));
  PetscCall(VecDestroy(&gms->r));
  PetscCall(VecDestroy(&gms->work_n));
  PetscCall(VecDestroy(&gms->work_n2));
  PetscCall(MatDestroy(&gms->V0));
  PetscCall(MatDestroy(&gms->V1));
  PetscCall(PetscFree(gms->Z));

  PetscCall(VecDuplicate(b, &gms->b_local));
  PetscCall(VecDuplicate(b, &gms->x_global));
  PetscCall(VecDuplicate(b, &gms->r));

  /* xGlobal := initial guess (which is x_local on entry; if user provided
     KSPSetInitialGuessNonzero this is non-zero). */
  PetscCall(VecCopy(x_local, gms->x_global));

  /* Phase 4a — for PC_RIGHT (and PC_SYMMETRIC, deferred), save the user's
     initial guess in a dedicated Vec. The algorithm internally tracks
     x_alg = x_global + x_local in the M-space (where M = A·B⁻¹). At solve
     end the user-visible solution is recovered as
     x_user = x_initial + B⁻¹·(x_alg − x_initial). Without this save we
     can't recover x_user after the algorithm has accumulated restart
     contributions into x_global.

     Done AFTER the per-solve VecDestroy block above (so a stale Vec from
     a prior solve is freed) and BEFORE the VecSet(x_local, 0.0) below
     (so the initial guess is still readable from x_global / x_local). */
  {
    PCSide pc_side_for_init;
    PetscCall(KSPGetPCSide(ksp, &pc_side_for_init));
    if (pc_side_for_init == PC_RIGHT || pc_side_for_init == PC_SYMMETRIC) {
      PetscCall(VecDuplicate(b, &gms->x_initial_guess));
      PetscCall(VecCopy(gms->x_global, gms->x_initial_guess));
    }
    /* For PC_NONE / PC_LEFT, gms->x_initial_guess stays NULL — the
       algorithm-internal x_alg coincides with x_user, and the destroy
       block above guarantees there's no stale Vec lurking. */
  }

  /* Constructor snapshot — matches the C++ port's PerfMeasure constructor:
     iter=0, matvec=0, iterres=||b||, trueres=||b||, runtime=0.

     Routed through Snapshot_Private so KSPConvergedDefault is called with
     n=0 and properly initializes ksp->rnorm0 (otherwise rnorm0 stays 0
     and every subsequent residual triggers a spurious DIVERGED_DTOL).
     With x_global=initial guess and x_local=0 here, x_total=x_global, so
     Snapshot_Private will compute trueres = ||b - A*x_global||. For zero
     initial guess (the bit-equivalence harness), this equals ||b||
     exactly. */
  {
    PetscReal nb;
    PetscCall(VecNorm(b, NORM_2, &nb));
    PetscCall(KSPGMSTABSnapshot_Private(ksp, gms, gms->x_global, nb));
  }

  /* If KSPConvergedDefault flagged convergence on the constructor row
     (e.g. ||b|| ~ 0 or already below abstol), respect that immediately —
     ws hasn't been allocated yet, so no inner-workspace cleanup needed. */
  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /* Allocate inner GMRES workspace sized for the largest cycle (m_max = 2s+2). */
  KSPGMSTABInnerWorkspace ws;
  PetscCall(KSPGMSTABInnerWorkspaceCreate_Private(ksp, gms->s, 2 * gms->s + 2, b, &ws));
  ws.matvec_count_ptr = &gms->matvec_count;

  /* bLocal = b - A * x_initial (counted matvec).
     CRUCIAL: this must be the UNPRECONDITIONED residual (one bare A
     application, NOT KSP_PCApplyBAorAB). The algorithm's M-space
     residual algebra is set up so that
         r = bLocal - M·x_alg = b - A·x_user
     which only holds if bLocal = b - A·x_initial. Using
     KSP_PCApplyBAorAB here would compute b - M·x_initial, leaving the
     residual permanently offset by `A·(I − B⁻¹)·x_initial` — invisible
     when x_initial = 0 (the bit-equivalence harness) but catastrophic
     for KSPSetInitialGuessNonzero with PC_RIGHT, as Phase 4a's
     ex_gmstab_pcright_nzg tripwire demonstrated.
     For PC_NONE this is identical to the previous behavior since
     KSP_PCApplyBAorAB(PC_NONE) is just MatMult(A). */
  Mat Amat;
  PetscCall(KSPGetOperators(ksp, &Amat, NULL));
  Vec Ax;
  PetscCall(VecDuplicate(b, &Ax));
  PetscCall(MatMult(Amat, gms->x_global, Ax));
  gms->matvec_count++;
  PetscCall(VecWAXPY(gms->b_local, -1.0, Ax, b));
  PetscCall(VecDestroy(&Ax));

  /* Phase 4b — for PC_LEFT we solve B⁻¹·A·x = B⁻¹·b. Convert bLocal to
     its preconditioned form ONCE here; from this point on the algorithm
     operates in B⁻¹·A's M-space and tracks r_pre = B⁻¹·(b − A·x_user).
     The cycles already use KSP_PCApplyBAorAB which dispatches B⁻¹·A on
     PC_LEFT, so r := bLocal_pre − M·x_local stays consistent with
     B⁻¹·(b − A·x_user) throughout the solve.

     The PCApply is uncounted (it's PC setup-equivalent work, not an
     A-matvec). Done after bLocal is fully formed and before VecCopy
     into gms->r below.

     For PC_NONE / PC_RIGHT / PC_SYMMETRIC this block is a no-op. */
  {
    PCSide pc_side_for_blocal;
    PetscCall(KSPGetPCSide(ksp, &pc_side_for_blocal));
    if (pc_side_for_blocal == PC_LEFT) {
      Vec b_pre;
      PetscCall(VecDuplicate(b, &b_pre));
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCApply(pc, gms->b_local, b_pre));
      PetscCall(VecCopy(b_pre, gms->b_local));
      PetscCall(VecDestroy(&b_pre));
    }
  }

  PetscCall(VecCopy(gms->b_local, gms->r));
  PetscCall(VecSet(x_local, 0.0));
  PetscCall(VecNorm(gms->r, NORM_2, &gms->beta));
  gms->beta_local = gms->beta;
  gms->beta_max   = gms->beta;
  PetscReal beta_curr = gms->beta;

  /* Initialisation (cold start). */
  PetscCall(KSPGMSTABInitialisation_Private(ksp, gms, &ws, x_local, gms->r, &beta_curr));
  gms->beta = beta_curr;

  /* Post-init duplicate snapshot — matches C++ solver.cpp:743. Emitted
     UNCONDITIONALLY here so the trace structure matches C++ even on
     paths where Init bails early (e.g. its inner gmres converged). The
     payload is unchanged from Init's internal final perf.read (same x,
     same beta, same matvec count); the row is a deliberate duplicate. */
  PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));

  if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) {
    /* Convergence test inside Initialisation OR the post-init dup
       triggered. After this point x_local has been zeroed and
       accumulates the cycle-local update. FinalizeSolution_Private
       turns x_local into the user-visible solution: for PC_NONE/PC_LEFT
       this is x_global + x_local; for PC_RIGHT it adds the B⁻¹
       unwrap. (For zero initial guess + PC_NONE this is the
       previous-Phase-3d VecAXPY no-op; for nonzero initial guess +
       PC_NONE it restores the KSPSetInitialGuessNonzero contract; for
       PC_RIGHT it produces the correct x_user via Phase 4a's algebra.) */
    PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
    PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (gms->beta <= ksp->abstol) {
    ksp->reason = KSP_CONVERGED_ATOL;
    PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
    PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Phase 3b validation path: -ksp_gmstab_force_l1_only runs exactly one
     Cycle1 invocation, then exits. Lets ex_gmstab_cycle1 diff the cycle's
     output against the C++ oracle without the flying-restart driver in
     the way. The post-init dup snapshot fired above plays the role of
     the C++ "begin-of-cycle" perf.read for this path too. */
  if (gms->force_l1_only) {
    PetscCall(KSPGMSTABCycle1_Private(ksp, gms, &ws, x_local, gms->r, &beta_curr));
    gms->beta = beta_curr;
    gms->cycle_count++;

    /* Post-cycle snapshot — matches the C++ main solver loop's perf.read
       at solver.cpp:599-602, paired with the cycle's own final read. */
    PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));

    if (gms->beta <= ksp->abstol) {
      ksp->reason = KSP_CONVERGED_ATOL;
    } else if (!ksp->reason) {
      /* Single-cycle validation: nothing more to do, even if beta > tolabs. */
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp,
        "KSPSolve_GMSTAB force_l1_only: ran one Cycle1, beta=%.6e > tolabs=%.6e\n",
        (double)gms->beta, (double)ksp->abstol));
    }
    PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
    PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (gms->force_l2_only) {
    /* Symmetric to force_l1_only: post-init dup served as begin-of-cycle,
       one Cycle2, post-cycle snapshot, exit. */
    PetscCall(KSPGMSTABCycle2_Private(ksp, gms, &ws, x_local, gms->r, &beta_curr));
    gms->beta = beta_curr;
    gms->cycle_count++;
    gms->n2cycles++;

    PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));

    if (gms->beta <= ksp->abstol) {
      ksp->reason = KSP_CONVERGED_ATOL;
    } else if (!ksp->reason) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;
      PetscCall(PetscInfo(ksp,
        "KSPSolve_GMSTAB force_l2_only: ran one Cycle2, beta=%.6e > tolabs=%.6e\n",
        (double)gms->beta, (double)ksp->abstol));
    }
    PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));
    PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* ============================================================================
     Natural-flow flying-restart driver loop. 1:1 port of solver.cpp:742-815.

     Snapshot rhythm:
       - Init's internal final snapshot fired inside Initialisation.
       - The post-init duplicate snapshot fired UNCONDITIONALLY above
         (matches C++ line 743); see the unified snapshot site after Init
         returns. We don't emit it again here.
       - Each loop iter: cycle (which emits its own internal final snapshot),
         then optional restart/replace work (with one counted matvec), then
         an end-of-iter snapshot (matches C++ line 812).

     betaMax bookkeeping: post-init betaMax = max(post-init beta, betaLocal).
     C++ does this at solver.cpp:746 — we mirror it AFTER Initialisation has
     run and updated beta_curr. (Note: gms->beta_max was set above to the
     pre-init beta as a placeholder for the force_l*_only paths, which
     don't touch betaMax — here we overwrite it with the correct post-init
     value.) ============================================================================ */

  gms->beta_max = PetscMax(beta_curr, gms->beta_local);

  PetscCheck(gms->s <= 64, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTAB: s = %" PetscInt_FMT " exceeds the natural-flow driver's "
             "stack-allocated eta/xi buffer (max 64). Bump the buffer or set s smaller.",
             gms->s);

  Vec Ax_drv;
  PetscCall(VecDuplicate(b, &Ax_drv));   /* scratch for the counted matvec at restart/replace */

  /* Loop continues until KSPConvergedDefault sets ksp->reason (via the
     snapshots inside the cycle / dir_rbio / end-of-iter), or we hit some
     other termination set externally on ksp->reason.

     IMPORTANT (Phase 4b): the loop is REASON-driven, not beta-driven. The
     algorithm-internal beta_curr is the *natural* residual of the
     preconditioned operator: ||r_unpre|| for PC_NONE / PC_RIGHT (those
     two algebras leave the tracked r in unpreconditioned form), but
     ||r_pre|| for PC_LEFT. The user's tolerance applies to the residual
     they asked for via KSPSetNormType (UNPRECONDITIONED → ||r_unpre||).
     Snapshot_Private's call to (*ksp->converged) does that check on the
     CORRECT residual for the user's normtype; the driver's job is just
     to honor the resulting ksp->reason.

     A previous version of this loop also early-broke on
     `beta_curr <= ksp->abstol`. That short-circuit is correct for
     PC_NONE / PC_RIGHT (where beta_curr ~ user-visible residual) but
     incorrect for PC_LEFT (where ||r_pre|| can be << ||r_unpre|| under
     a well-conditioned preconditioner — Phase 4b's pcleft_jacobi
     tripwire caught the regression before this fix). The right safety
     net is `if (ksp->reason && reason != ITERATING) break`, which fires
     uniformly across all PC sides. */
  /* Phase 4 audit follow-up: the natural-flow loop is reason-driven.
     Snapshot_Private now properly enforces ksp->max_it (sets DIVERGED_ITS
     when ksp->its >= ksp->max_it; the standard PETSc convention that
     KSPConvergedDefault does NOT do itself). Each cycle emits multiple
     snapshots, so DIVERGED_ITS fires within a fraction-of-a-cycle of the
     user's max_it threshold. No additional cycle-level cap needed. */
  while (!ksp->reason || ksp->reason == KSP_CONVERGED_ITERATING) {
    PetscBool t_restart = PETSC_FALSE, t_replace = PETSC_FALSE;
    if (beta_curr < gms->c_restart * gms->beta_local) {
      t_restart = PETSC_TRUE;
    } else if (beta_curr < gms->c_replace * gms->beta_max) {
      t_replace = PETSC_TRUE;
    } else {
      gms->beta_max = PetscMax(gms->beta_max, beta_curr);
    }

    /* L selection (C++ solver.cpp:772-775). force_l1_only / force_l2_only are
       handled by the early-out branches above and never reach here. */
    PetscInt L_eff = (t_restart || t_replace || gms->n2cycles > gms->n2cycles_max) ? 1 : 2;

    if (L_eff == 1) {
      PetscCall(KSPGMSTABCycle1_Private(ksp, gms, &ws, x_local, gms->r, &beta_curr));
      gms->n2cycles = 0;
    } else {
      PetscCall(KSPGMSTABCycle2_Private(ksp, gms, &ws, x_local, gms->r, &beta_curr));
      gms->n2cycles += 1;
    }
    gms->beta = beta_curr;
    gms->cycle_count++;

    /* Cycle's internal final snapshot may have triggered convergence /
       divergence via KSPConvergedDefault. Honour that. */
    if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) break;

    if (t_restart || t_replace) {
      /* r = bLocal - A * x_local;  counted matvec (C++ solver.cpp:795). */
      PetscCall(KSP_PCApplyBAorAB(ksp, x_local, Ax_drv, ws.work_n));
      gms->matvec_count++;
      PetscCall(VecWAXPY(gms->r, -1.0, Ax_drv, gms->b_local));

      /* eta = P^T * r — s VecDots in P-column order (matches Eigen's
         column-major P.transpose() * r). */
      PetscScalar eta[64];
      for (PetscInt k = 0; k < gms->s; ++k) {
        Vec pk;
        PetscCall(MatDenseGetColumnVecRead(gms->P, k, &pk));
        PetscCall(VecDot(gms->r, pk, &eta[k]));
        PetscCall(MatDenseRestoreColumnVecRead(gms->P, k, &pk));
      }

      /* dir_rbio: xi := Z \ eta (lower-tri); r -= V1 * xi; x += V0 * xi.
         Mirror of gmstab_cpp::dir_rbio. The trsv writes xi into eta[] in
         place. The two AXPY loops walk V1 and V0 columns simultaneously
         to amortize column borrow/restore. */
      PetscCall(KSPGMSTABTrsv_Private("L", "N", "N", gms->s, gms->Z, gms->Z_ldim, eta, 1));
      for (PetscInt k = 0; k < gms->s; ++k) {
        Vec V1k, V0k;
        PetscCall(MatDenseGetColumnVecRead(gms->V1, k, &V1k));
        PetscCall(VecAXPY(gms->r, -eta[k], V1k));
        PetscCall(MatDenseRestoreColumnVecRead(gms->V1, k, &V1k));
        PetscCall(MatDenseGetColumnVecRead(gms->V0, k, &V0k));
        PetscCall(VecAXPY(x_local, eta[k], V0k));
        PetscCall(MatDenseRestoreColumnVecRead(gms->V0, k, &V0k));
      }
      PetscCall(VecNorm(gms->r, NORM_2, &beta_curr));
      gms->beta = beta_curr;
      gms->beta_max = beta_curr;
    } else {
      gms->beta_max = PetscMax(beta_curr, gms->beta_max);
    }

    if (t_restart) {
      /* C++ solver.cpp:804-808:  bLocal = r;  xGlobal += x;  x = 0;
                                   betaLocal = beta. */
      PetscCall(VecCopy(gms->r, gms->b_local));
      PetscCall(VecAXPY(gms->x_global, 1.0, x_local));
      PetscCall(VecSet(x_local, 0.0));
      gms->beta_local = beta_curr;
    }

    /* End-of-iter snapshot (C++ solver.cpp:812). */
    PetscCall(KSPGMSTABSnapshotLocal_Private(ksp, gms, x_local, beta_curr));
    if (ksp->reason && ksp->reason != KSP_CONVERGED_ITERATING) break;
  }

  /* The loop is reason-driven (Phase 4b refactor). If we exit with
     ksp->reason still ITERATING, something unusual happened — typically
     a KSPConvergedDefault decision that didn't classify (shouldn't be
     possible, but defensive). Set DIVERGED_BREAKDOWN so the user sees a
     diagnostic rather than a silent ITERATING-but-returned. */
  if (!ksp->reason || ksp->reason == KSP_CONVERGED_ITERATING) {
    ksp->reason = KSP_DIVERGED_BREAKDOWN;
    PetscCall(PetscInfo(ksp,
      "KSPSolve_GMSTAB: natural-flow loop exited with ksp->reason=ITERATING "
      "(unexpected — KSPConvergedDefault should always classify); reporting "
      "DIVERGED_BREAKDOWN. Last beta_curr=%.6e tolabs=%.6e\n",
      (double)beta_curr, (double)ksp->abstol));
  }

  /* On exit, leave the user-visible solution in ksp->vec_sol.
     FinalizeSolution_Private subsumes the historical
     `VecAXPY(x_local, 1.0, gms->x_global)` accumulation and additionally
     applies the right-preconditioning unwrap when pc_side == PC_RIGHT
     (Phase 4a). For PC_NONE this is identical to the prior behavior. */
  PetscCall(KSPGMSTABFinalizeSolution_Private(ksp, gms, x_local));

  PetscCall(VecDestroy(&Ax_drv));
  PetscCall(KSPGMSTABInnerWorkspaceDestroy_Private(&ws));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPSetUp — Phase 1: validate options and fail fast on misuse.
 *
 * Vector / Mat allocation comes in Phase 3 once we know exactly how many
 * scratch vectors the cycle bodies need. For Phase 1 we just sanity-check
 * the shadow-space configuration so misuse is caught early.
 * ============================================================================ */
static PetscErrorCode KSPSetUp_GMSTAB(KSP ksp)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(gms->s >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTAB: shadow-space dimension s = %" PetscInt_FMT
             " must be >= 1 (set with -ksp_gmstab_s)",
             gms->s);
  PetscCheck(gms->L == 1 || gms->L == 2, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTAB: L = %" PetscInt_FMT " must be 1 or 2 (set with -ksp_gmstab_L)",
             gms->L);
  PetscCheck(gms->c_restart > 0.0 && gms->c_restart < 1.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTAB: c_restart = %g must be in (0, 1)", (double)gms->c_restart);
  PetscCheck(gms->c_replace > 0.0 && gms->c_replace < 1.0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTAB: c_replace = %g must be in (0, 1)", (double)gms->c_replace);

  /* Shadow space P (will actually be loaded / built in Phase 3 or Phase 5
     on the first iteration; here we only validate that no two sources
     contradict each other). */
  if (gms->P_user) {
    Mat      Amat = NULL, Pmat = NULL;
    PetscInt N_op, M_op;
    PetscInt M_p, N_p;
    PetscCall(KSPGetOperators(ksp, &Amat, &Pmat));
    PetscCall(MatGetSize(Amat, &M_op, &N_op));
    PetscCall(MatGetSize(gms->P_user, &M_p, &N_p));
    PetscCheck(M_p == M_op, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP,
               "KSPGMSTAB: user-set shadow space P has %" PetscInt_FMT
               " rows but operator has %" PetscInt_FMT,
               M_p, M_op);
    PetscCheck(N_p == gms->s, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP,
               "KSPGMSTAB: user-set shadow space P has %" PetscInt_FMT
               " cols but s = %" PetscInt_FMT,
               N_p, gms->s);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPReset — release all per-solve state.
 *
 * Frees the persistent Mats (V0, V1, W, hU, P), Vecs (r, work_n, work_n2,
 * x_global, b_local), the host-side Z buffer, and any open trace file.
 * Does NOT free P_user (that's a borrowed reference) or static config
 * fields like P_file (which is just a fixed-size char array).
 * ============================================================================ */
static PetscErrorCode KSPReset_GMSTAB(KSP ksp)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&gms->V0));
  PetscCall(MatDestroy(&gms->V1));
  PetscCall(MatDestroy(&gms->W));
  PetscCall(MatDestroy(&gms->P));
  PetscCall(MatDestroy(&gms->hU));
  PetscCall(VecDestroy(&gms->r));
  PetscCall(VecDestroy(&gms->work_n));
  PetscCall(VecDestroy(&gms->work_n2));
  PetscCall(VecDestroy(&gms->x_global));
  PetscCall(VecDestroy(&gms->x_initial_guess));
  PetscCall(VecDestroy(&gms->b_local));
  PetscCall(PetscFree(gms->Z));
  if (gms->trace_fp) {
    fclose(gms->trace_fp);
    gms->trace_fp = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPDestroy — release everything Reset releases plus the user-set P
 * reference, the random generator, and the function-table hooks for the
 * public setters. KSPDestroyDefault frees ksp->data.
 * ============================================================================ */
static PetscErrorCode KSPDestroy_GMSTAB(KSP ksp)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCall(KSPReset_GMSTAB(ksp));
  PetscCall(MatDestroy(&gms->P_user));
  PetscCall(PetscRandomDestroy(&gms->prand));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMSTABSetS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMSTABGetS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMSTABSetShadowSpace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGMSTABSetShadowSpaceFile_C", NULL));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPView — pretty-print the configured solver knobs.
 * ============================================================================ */
static PetscErrorCode KSPView_GMSTAB(KSP ksp, PetscViewer viewer)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;
  PetscBool   isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  s (shadow-space dim)             = %" PetscInt_FMT "\n", gms->s));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  L (poly stab degree)             = %" PetscInt_FMT "\n", gms->L));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  c_restart (full-restart trigger) = %g\n", (double)gms->c_restart));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  c_replace (replace-only trigger) = %g\n", (double)gms->c_replace));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  n2cycles_max                     = %" PetscInt_FMT "\n", gms->n2cycles_max));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  stab_angle (rad / deg)           = %g / %g\n",
              (double)gms->stab_angle, (double)(gms->stab_angle * 180.0 / M_PI)));
    if (gms->P_user) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  shadow space P                   = user-supplied via KSPGMSTABSetShadowSpace\n"));
    } else if (gms->P_file[0]) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  shadow space P                   = file %s\n", gms->P_file));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  shadow space P                   = mt19937_64 (seed %lu) + orth (default)\n",
                                       gms->rng_seed));
    }
    if (gms->has_recycling) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  recycling enabled, tolabs2 = %g\n", (double)gms->tolabs2));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  recycling                        = off\n"));
    }
    if (gms->trace_csv) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  per-iter trace CSV               = %s\n", gms->trace_csv_path));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Public setter: KSPGMSTABSetS
 * ============================================================================ */
static PetscErrorCode KSPGMSTABSetS_GMSTAB(KSP ksp, PetscInt s)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(s >= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
             "KSPGMSTABSetS: s = %" PetscInt_FMT " must be >= 1", s);
  if (gms->s != s) {
    gms->s          = s;
    ksp->setupstage = KSP_SETUP_NEW;
    /* Drop user-set P if its column count no longer matches s. */
    if (gms->P_user) {
      PetscInt N_p;
      PetscCall(MatGetSize(gms->P_user, NULL, &N_p));
      if (N_p != s) PetscCall(MatDestroy(&gms->P_user));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGMSTABSetS - Set the shadow-space dimension s for `KSPGMSTAB`.

  Logically Collective

  Input Parameters:
+ ksp - iterative context
- s   - shadow-space dimension (>= 1; default 4; canonical IDR(s) values 1..8)

  Options Database Key:
. -ksp_gmstab_s s - shadow-space dimension

  Level: intermediate

.seealso: `KSPGMSTAB`, `KSPGMSTABGetS`, `KSPGMSTABSetShadowSpace`
@*/
PetscErrorCode KSPGMSTABSetS(KSP ksp, PetscInt s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, s, 2);
  PetscTryMethod(ksp, "KSPGMSTABSetS_C", (KSP, PetscInt), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPGMSTABGetS_GMSTAB(KSP ksp, PetscInt *s)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;
  PetscFunctionBegin;
  *s = gms->s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGMSTABGetS - Get the shadow-space dimension s currently configured on a `KSPGMSTAB`.

  Not Collective

  Input Parameter:
. ksp - iterative context

  Output Parameter:
. s - shadow-space dimension

  Level: intermediate

.seealso: `KSPGMSTAB`, `KSPGMSTABSetS`
@*/
PetscErrorCode KSPGMSTABGetS(KSP ksp, PetscInt *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(s, 2);
  PetscUseMethod(ksp, "KSPGMSTABGetS_C", (KSP, PetscInt *), (ksp, s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Public setter: KSPGMSTABSetShadowSpace (programmatic P override)
 *
 * The Mat is reference-counted, so the caller can destroy their copy after
 * the call. PETSc owns it from this point.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABSetShadowSpace_GMSTAB(KSP ksp, Mat P)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)P));
  PetscCall(MatDestroy(&gms->P_user));
  gms->P_user      = P;
  gms->P_file[0]   = '\0';   /* file path takes lower precedence; clear it */
  ksp->setupstage  = KSP_SETUP_NEW;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGMSTABSetShadowSpace - Programmatically set the GMstab shadow space matrix P (column-major dense, N x s).

  Logically Collective

  Input Parameters:
+ ksp - iterative context
- P   - dense matrix, M(P) = M(operator), N(P) = s

  Notes:
  P is reference-counted; the caller may destroy their handle after the
  call. The shadow space is reset whenever `KSPGMSTABSetS()` changes s
  to a value that doesn't match P's column count.

  This is the recommended path for the bit-equivalence validation harness:
  the validator reads MATLAB's exported `P.bin` into a `MATDENSE` and hands
  it here so the PETSc port runs against the same shadow space MATLAB used.

  Level: advanced

.seealso: `KSPGMSTAB`, `KSPGMSTABSetShadowSpaceFile`, `KSPGMSTABSetS`
@*/
PetscErrorCode KSPGMSTABSetShadowSpace(KSP ksp, Mat P)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(P, MAT_CLASSID, 2);
  PetscTryMethod(ksp, "KSPGMSTABSetShadowSpace_C", (KSP, Mat), (ksp, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Public setter: KSPGMSTABSetShadowSpaceFile
 *
 * The path is stashed; the file is actually loaded in KSPSetUp_GMSTAB
 * (Phase 3) once we know N from ksp->pmat.
 * ============================================================================ */
static PetscErrorCode KSPGMSTABSetShadowSpaceFile_GMSTAB(KSP ksp, const char *path)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;

  PetscFunctionBegin;
  PetscCall(PetscStrncpy(gms->P_file, path, sizeof(gms->P_file)));
  ksp->setupstage = KSP_SETUP_NEW;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  KSPGMSTABSetShadowSpaceFile - Set the path to a binary file holding the GMstab shadow space P.

  Logically Collective

  Input Parameters:
+ ksp  - iterative context
- path - file path. The file must contain N*s float64 values, column-major,
         little-endian, no header — same layout the validation package's
         P.bin files use.

  Options Database Key:
. -ksp_gmstab_p_file <path> - same effect via the options database

  Level: advanced

.seealso: `KSPGMSTAB`, `KSPGMSTABSetShadowSpace`, `KSPGMSTABSetS`
@*/
PetscErrorCode KSPGMSTABSetShadowSpaceFile(KSP ksp, const char *path)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(path, 2);
  PetscTryMethod(ksp, "KSPGMSTABSetShadowSpaceFile_C", (KSP, const char *), (ksp, path));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * KSPSetFromOptions
 * ============================================================================ */
static PetscErrorCode KSPSetFromOptions_GMSTAB(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_GMSTAB *gms = (KSP_GMSTAB *)ksp->data;
  PetscInt    this_s;
  PetscReal   this_real;
  PetscInt    this_int;
  PetscBool   flg;
  char        path[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSPGMSTAB Options");

  PetscCall(PetscOptionsInt("-ksp_gmstab_s",
            "shadow-space dimension (>= 1; default 4)",
            "KSPGMSTABSetS", gms->s, &this_s, &flg));
  if (flg) PetscCall(KSPGMSTABSetS(ksp, this_s));

  PetscCall(PetscOptionsInt("-ksp_gmstab_L",
            "polynomial stabilization degree (1 or 2)",
            NULL, gms->L, &this_int, &flg));
  if (flg) {
    PetscCheck(this_int == 1 || this_int == 2,
               PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE,
               "-ksp_gmstab_L = %" PetscInt_FMT " must be 1 or 2", this_int);
    gms->L          = this_int;
    ksp->setupstage = KSP_SETUP_NEW;
  }

  PetscCall(PetscOptionsReal("-ksp_gmstab_c_restart",
            "flying-restart trigger (full restart when beta < c_restart * betaLocal)",
            NULL, gms->c_restart, &this_real, &flg));
  if (flg) gms->c_restart = this_real;

  PetscCall(PetscOptionsReal("-ksp_gmstab_c_replace",
            "flying-restart trigger (replace when beta < c_replace * betaMax)",
            NULL, gms->c_replace, &this_real, &flg));
  if (flg) gms->c_replace = this_real;

  PetscCall(PetscOptionsInt("-ksp_gmstab_n2cycles_max",
            "max consecutive Cycle-2 cycles before forcing Cycle-1 (default 3)",
            NULL, gms->n2cycles_max, &this_int, &flg));
  if (flg) gms->n2cycles_max = this_int;

  /* Phase 3b validation knobs — run exactly one Cycle{1,2} after
     Initialisation and exit. Used by ex_gmstab_cycle{1,2} to validate
     the cycle's algebra against the C++ oracle in isolation. Off in
     production. */
  {
    PetscBool this_bool;
    PetscCall(PetscOptionsBool("-ksp_gmstab_force_l1_only",
              "validation: run exactly ONE Cycle1 after Initialisation, then exit BREAKDOWN",
              NULL, gms->force_l1_only, &this_bool, &flg));
    if (flg) gms->force_l1_only = this_bool;

    PetscCall(PetscOptionsBool("-ksp_gmstab_force_l2_only",
              "validation: run exactly ONE Cycle2 after Initialisation, then exit BREAKDOWN",
              NULL, gms->force_l2_only, &this_bool, &flg));
    if (flg) gms->force_l2_only = this_bool;
  }

  /* Recycling — Phase 8 only. Knob is exposed but warning issued if used. */
  PetscCall(PetscOptionsReal("-ksp_gmstab_tolabs2",
            "recycling-dump threshold (Phase 8 only; ignored at present)",
            NULL, gms->tolabs2, &this_real, &flg));
  if (flg) {
    gms->tolabs2 = this_real;
    if (PetscIsNormalReal(this_real) && this_real != PETSC_INFINITY) {
      PetscCall(PetscInfo(ksp,
        "KSPGMSTAB: -ksp_gmstab_tolabs2 set but recycling is unimplemented "
        "until Phase 8; the value is stored but has no effect.\n"));
    }
  }

  /* Shadow-space path */
  path[0] = '\0';
  PetscCall(PetscOptionsString("-ksp_gmstab_p_file",
            "path to a binary shadow-space file (N*s float64 little-endian)",
            "KSPGMSTABSetShadowSpaceFile",
            gms->P_file, path, sizeof(path), &flg));
  if (flg) PetscCall(KSPGMSTABSetShadowSpaceFile(ksp, path));

  /* Per-iter trace CSV (validation aid, off by default) */
  path[0] = '\0';
  PetscCall(PetscOptionsString("-ksp_gmstab_trace_csv",
            "emit per-snapshot CSV (iter,matvec,iterres,trueres,runtime,runtime_mv)",
            NULL, gms->trace_csv_path, path, sizeof(path), &flg));
  if (flg) {
    PetscCall(PetscStrncpy(gms->trace_csv_path, path, sizeof(gms->trace_csv_path)));
    gms->trace_csv = PETSC_TRUE;
  }

  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ============================================================================
 * Documentation block — picked up by petsc-doc / man-page generation.
 * ============================================================================ */

/*MC
  KSPGMSTAB - Implements GM(s)stab, a member of the IDR(s) family of Krylov
  subspace solvers for nonsymmetric linear systems.

  GM(s)stab differs from plain IDR(s)stab in three structural ways:

  1.  Flying restart between L=1 and L=2 stabilization cycles, automatically
      chosen at each outer step based on the residual-norm trajectory. The
      switch-over keeps the basis vectors well-conditioned without the user
      having to schedule restarts.
  2.  Internal use of modified GMRES variants (pGMRESm and augGMRESm) that
      have been projected against the current shadow space. This gives
      GM(s)stab a polynomial that minimizes the residual over the right
      Krylov subspace at every outer step (which IDR(s)stab does not do).
  3.  Subspace recycling (`hU`) for sequence-of-systems applications.
      Disabled by default; enable via -ksp_gmstab_tolabs2 (Phase 8 — not
      yet implemented as of the initial KSPGMSTAB integration).

  The polynomial step uses a "maintaining-the-convergence" strategy with a
  35° angle threshold: when the natural minimum-residual step would produce
  a tiny ω that destabilizes the next IDR step, the angle is forced up to
  the threshold.

  Options Database Keys:
+ -ksp_gmstab_s <int>          - shadow-space dimension (>= 1; default 4; canonical 1..8)
. -ksp_gmstab_L <1|2>           - polynomial stabilization degree (default 2; auto-selected)
. -ksp_gmstab_c_restart <real>  - full-restart trigger (default 1e-2)
. -ksp_gmstab_c_replace <real>  - replace-only trigger (default 1e-2)
. -ksp_gmstab_n2cycles_max <int> - max consecutive Cycle-2 (default 3)
. -ksp_gmstab_p_file <path>     - shadow-space file (N*s float64, col-major, little-endian)
. -ksp_gmstab_tolabs2 <real>    - recycling-dump threshold (Phase 8)
- -ksp_gmstab_trace_csv <path>  - emit per-snapshot CSV (validation aid)

  Level: intermediate

  Notes:
  KSPGMSTAB tracks the true residual `r = b - Ax` internally — unusually
  for the IDR(s) family. This makes right preconditioning (`KSP_PC_RIGHT`)
  the natural default, and allows `-ksp_norm_type unpreconditioned` to be
  served without an extra `PCApply` per iteration. Left and split (symmetric)
  preconditioning are also supported (with one extra `PCApply` per residual
  update for true-residual recovery).

  Programmatic shadow-space override is available via `KSPGMSTABSetShadowSpace()`
  for the validation harness.

.seealso: `KSP`, `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSPBCGS`, `KSPBCGSL`,
          `KSPGMSTABSetS()`, `KSPGMSTABSetShadowSpace()`, `KSPGMSTABSetShadowSpaceFile()`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_GMSTAB(KSP ksp)
{
  KSP_GMSTAB *gms;

  PetscFunctionBegin;
  PetscCall(PetscNew(&gms));
  ksp->data = (void *)gms;

  /* Defaults — match the C++ port and MATLAB reference. */
  gms->s             = 4;
  gms->L             = 2;
  gms->c_restart     = (PetscReal)1e-2;
  gms->c_replace     = (PetscReal)1e-2;
  gms->n2cycles_max  = 3;
  gms->force_l1_only = PETSC_FALSE;
  gms->force_l2_only = PETSC_FALSE;
  gms->stab_angle    = KSPGMSTAB_DEFAULT_STAB_ANGLE;
  gms->tolabs2       = PETSC_INFINITY;
  gms->has_recycling = PETSC_FALSE;
  gms->rng_seed      = KSPGMSTAB_DEFAULT_SEED;

  /* Per-iter state defaults — Reset / SetUp will populate these properly. */
  gms->matvec_count   = 0;
  gms->snapshot_count = 0;

  /* Supported PC sides x norm types.
     Priority: PC_RIGHT + UNPRECONDITIONED is the canonical / fastest path
     because the algorithm tracks the true residual internally.

     PC_SYMMETRIC is intentionally NOT declared here. The helper code at
     gmstab_helpers.c:68/163 currently treats PC_SYMMETRIC as a synonym for
     PC_RIGHT (uses PCApply for the unwrap), but that's wrong for split
     symmetric preconditioners B = B_L * B_R — the unwrap should apply
     B_R⁻¹ via PCApplySymmetricRight. Until Phase 4c implements that
     properly, we reject PC_SYMMETRIC at setup by not registering it; users
     who set KSPSetPCSide(ksp, PC_SYMMETRIC) will get a clean PETSc error
     ("KSPGMSTAB does not support that side / norm combination") instead of
     a silent wrong-answer. Matches GMRES's convention for unsupported
     sides. */
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED,   PC_RIGHT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT,  2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED,   PC_LEFT,  3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE,             PC_RIGHT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE,             PC_LEFT,  1));

  ksp->ops->setup          = KSPSetUp_GMSTAB;
  ksp->ops->solve          = KSPSolve_GMSTAB;
  ksp->ops->reset          = KSPReset_GMSTAB;
  ksp->ops->destroy        = KSPDestroy_GMSTAB;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_GMSTAB;
  ksp->ops->view           = KSPView_GMSTAB;

  /* Compose public setter functions — these dispatch through KSPGMSTABSet*
     wrappers via PetscTryMethod. */
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,
            "KSPGMSTABSetS_C",
            KSPGMSTABSetS_GMSTAB));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,
            "KSPGMSTABGetS_C",
            KSPGMSTABGetS_GMSTAB));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,
            "KSPGMSTABSetShadowSpace_C",
            KSPGMSTABSetShadowSpace_GMSTAB));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp,
            "KSPGMSTABSetShadowSpaceFile_C",
            KSPGMSTABSetShadowSpaceFile_GMSTAB));

  PetscFunctionReturn(PETSC_SUCCESS);
}
