/*
   Private data structure for KSPGMSTAB — a PETSc port of GM(s)stab, an
   IDR(s)-family Krylov solver with a "flying restart" between L=1 and L=2
   stabilization cycles, internal use of modified GMRES projected against
   a shadow space, and a maintaining-the-convergence (35°) angle threshold
   in its polynomial step.

   The C++ reference port lives at /home/sam/hpc_stack/gmstab_cpp; the
   MATLAB reference at /home/sam/hpc_stack/gmstab_matlab. Bit-for-bit
   baselines (with deterministic shadow space P) live in the validation
   package; Phase 5a of the implementation plan validates KSPGMSTAB
   against those baselines to <= 1e-10 absolute drift on every snapshot.

   See also:
   - src/ksp/ksp/impls/gmstab/IMPLEMENTATION_PLAN.md (design, phasing)
   - src/ksp/ksp/impls/gmstab/PHASE0_VERIFICATION.md (oracle confirmed)
*/
#pragma once
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

PETSC_INTERN PetscErrorCode KSPGMSTABSetShadowSpace(KSP, Mat);
PETSC_INTERN PetscErrorCode KSPGMSTABSetShadowSpaceFile(KSP, const char *);
PETSC_INTERN PetscErrorCode KSPGMSTABSetS(KSP, PetscInt);
PETSC_INTERN PetscErrorCode KSPGMSTABGetS(KSP, PetscInt *);

typedef struct {
  /* User-facing parameters (configured via -ksp_gmstab_*). All have
     sensible defaults set in KSPCreate_GMSTAB and overridable from
     KSPSetFromOptions_GMSTAB. */
  PetscInt    s;                /* shadow-space dimension (default 4) */
  PetscInt    L;                /* polynomial stabilization degree (informational
                                   for now; cycle dispatch is automatic per the
                                   flying-restart heuristic) */
  PetscReal   stab_angle;       /* maintaining-the-convergence threshold,
                                   default M_PI * 7 / 36 (35°), exposed
                                   only for completeness — not in
                                   the harness's option set */
  PetscReal   c_restart;        /* flying-restart trigger: full restart when
                                   beta < c_restart * betaLocal (default 1e-2) */
  PetscReal   c_replace;        /* flying-restart trigger: replace when
                                   beta < c_replace * betaMax (default 1e-2) */
  PetscInt    n2cycles_max;     /* force a Cycle1 after n2cycles_max
                                   consecutive Cycle2s (default 3) */
  PetscBool   force_l1_only;    /* validation knob (default PETSC_FALSE):
                                   when set, the driver runs Initialisation
                                   then exactly ONE Cycle1 and exits with
                                   KSP_DIVERGED_BREAKDOWN unless tolabs is
                                   reached. Used by ex_gmstab_cycle1 to
                                   isolate cycle 1 from the flying-restart
                                   driver (Phase 3b validation). */
  PetscBool   force_l2_only;    /* validation knob (default PETSC_FALSE):
                                   symmetric to force_l1_only. Runs
                                   Initialisation + exactly ONE Cycle2 +
                                   exits. If both force_l1_only and
                                   force_l2_only are set, force_l1_only
                                   wins (matches the C++ port's
                                   GMSTAB_FORCE_L1 vs GMSTAB_FORCE_L2
                                   precedence). */

  /* Recycling parameters (Phase 8 only — left at off / NULL until then) */
  PetscReal   tolabs2;          /* dump hU when iter-residual exceeds this;
                                   default = INFINITY (recycling off) */
  Mat         hU;               /* recycling subspace; NULL when off */
  PetscBool   has_recycling;

  /* Shadow space P
     ------------------------------------------------------------------------
     P is an N x s tall-skinny dense matrix. Three ways to provide it:

       (1) Programmatic:  KSPGMSTABSetShadowSpace(ksp, P)
                          User-owned Mat, Mat is PetscObjectReference()'d.
                          Used by the validation harness to inject MATLAB-
                          exported P.bin contents.

       (2) File-based:    -ksp_gmstab_p_file <path>     (loaded in SetUp)
                          Same expected layout as P.bin: N*s float64,
                          column-major little-endian, no header.

       (3) Default:       std::mt19937_64 + Gram-Schmidt orthonormalisation
                          (matches the C++ port's default_shadow_space()).
                          Used when production code doesn't care about
                          bit-equivalence with MATLAB.

     Precedence: (1) > (2) > (3).
  */
  Mat         P;                /* the actual N x s shadow space */
  Mat         P_user;           /* user-supplied via SetShadowSpace; refcount-held */
  char        P_file[PETSC_MAX_PATH_LEN];  /* path for option (2); empty if unset */
  PetscRandom prand;            /* lazily-allocated for option (3) */
  unsigned long rng_seed;       /* default 5489 (matches std::mt19937 default) */

  /* Persistent solver state across iterations.
     V0, V1, W are tall-skinny dense (N x s+1, N x s+1, N x m+1 where
     m = max(s, 2s+2) = 2s+2 when L=2 cycle is used).

     Z is the small s x s lower-triangular matrix maintained as the
     bi-orthogonality invariant Y = P^T * V1 = L_z * Q_z'. Stored as
     a dense host-side PetscScalar array (column-major). */
  Mat         V0, V1, W;
  PetscScalar *Z;               /* s x s, column-major, lower-triangular */
  PetscInt    Z_ldim;           /* leading dim of Z (= s for now) */

  /* Residual / norm tracking. The natural quantity is the TRUE residual
     r = b - A*x (i.e., GMstab tracks it explicitly, unlike most IDR(s)
     family solvers which track only the preconditioned residual). For
     left-preconditioned use, beta_pc tracks the preconditioned norm as
     well so KSP_NORM_PRECONDITIONED can be served without an extra
     PCApply per iteration. */
  Vec         r;                /* current residual, length N */
  Vec         work_n;           /* one length-N scratch */
  Vec         work_n2;          /* a second length-N scratch (used in
                                   pgmres / aug_gmres builds) */
  PetscReal   beta;             /* ||r||_2 (true residual norm) */
  PetscReal   beta_pc;          /* preconditioned residual norm (PC_LEFT
                                   only); 0 when not applicable */

  /* Driver state (flying restart) */
  Vec         x_global;         /* solution accumulator (xGlobal in C++ port);
                                   x_local lives in ksp->vec_sol */
  Vec         x_initial_guess;  /* user's initial guess at solve start —
                                   captured BEFORE x_local is zeroed. Only
                                   allocated when pc_side ∈ {PC_RIGHT,
                                   PC_SYMMETRIC}; for those modes the algorithm
                                   tracks x_alg = x_global + x_local in the
                                   M-space, and the user-visible solution is
                                   x_user = x_initial_guess + B⁻¹·(x_alg −
                                   x_initial_guess). For PC_NONE / PC_LEFT this
                                   field stays NULL — x_alg = x_user, no unwrap
                                   needed. Phase 4a. */
  Vec         b_local;          /* current right-hand side after restart */
  PetscReal   beta_local;       /* ||b_local|| at last restart */
  PetscReal   beta_max;         /* sup over residual norms since last full
                                   restart, for the c_replace trigger */
  PetscInt    cycle_count;      /* total cycles so far */
  PetscInt    n2cycles;         /* consecutive Cycle2 count */

  /* Validation tracing — emit a per-snapshot CSV in the
     iter,matvec,iterres,trueres,runtime,runtime_mv format the C++
     port produces. Used by Phase 5a's bit-equivalence harness. */
  PetscBool   trace_csv;
  char        trace_csv_path[PETSC_MAX_PATH_LEN];
  FILE       *trace_fp;
  PetscInt    snapshot_count;
  PetscInt    matvec_count;     /* counted matvecs (mirrors PerfMeasure) */
  PetscInt    _last_logged_matvec_count;  /* tripwire — last value matvec_count
                                             held at the previous Snapshot_Private
                                             call. Used to assert monotonicity
                                             across snapshots; set to 0 on each
                                             KSPSolve_GMSTAB entry. */
  PetscLogDouble t_total_start; /* wall-clock at last resume */
  PetscReal   t_total;          /* total runtime accumulated across pauses */
  PetscReal   t_mv;             /* runtime spent inside matvecs */
} KSP_GMSTAB;
