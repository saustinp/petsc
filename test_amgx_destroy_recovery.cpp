// Standalone reproducer + regression test for PETSc AMGX destructor bug.
//
// Without the fix to amgx.cxx::PCDestroy_AMGX, this program aborts via
// MPI_Abort with the "second error after first error" cascade when
// KSPDestroy is called on a KSP whose AMGX setup failed.
//
// With the fix, the program prints `[PASS]` and exits 0.
//
// Build:
//   cd /home/sam/hpc_stack/petsc/test_amgx_destroy_recovery_build || mkdir -p $_ && cd $_
//   /home/sam/.local/mpich/bin/mpicxx \
//       -std=c++17 \
//       -I/home/sam/hpc_stack/petsc/include \
//       -I/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/include \
//       /home/sam/hpc_stack/petsc/test_amgx_destroy_recovery.cpp \
//       -L/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/lib \
//       -Wl,-rpath,/home/sam/hpc_stack/petsc/arch-cuda-opt-i32/lib \
//       -lpetsc \
//       -o test_amgx_destroy_recovery
//
// Run:
//   ./test_amgx_destroy_recovery
//
// What it does:
//   1. Build a tiny 9x9 identity matrix.
//   2. Configure AMGX with `-pc_amgx_amg_method AGGREGATION` but DON'T
//      pass an explicit `-pc_amgx_selector`. The default (PMIS) is
//      invalid for AGGREGATION, so PCSetFromOptions_AMGX correctly
//      raises "Chosen selector is not used for AmgX Aggregation AMG".
//   3. Catch that error (good behavior — user code is allowed to recover).
//   4. Call KSPDestroy.
//      - PRE-PATCH: PCDestroy_AMGX hits PetscCheck(rsrc != nullptr) and
//        triggers PETSc's "second error" handler → MPI_Abort.
//      - POST-PATCH: destructor's null guards skip the rsrc/cfg/comm
//        teardown when nothing was allocated, returns cleanly.
//   5. Print PASS or FAIL.
//
// Returns:
//   0 on PASS (destructor cleanly handled partial init)
//   1 on FAIL (destructor errored)
//   Aborts (rc != 0 from MPI_Abort) if the bug is present and unfixed.

#include <petscksp.h>
#include <cstdio>

int main(int argc, char **argv)
{
    PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL,
        "AMGX PCDestroy on partial-init regression test");
    if (ierr) return ierr;

    // ---- Build a tiny identity matrix on host ----
    Mat A;
    PetscCall(MatCreate(PETSC_COMM_SELF, &A));
    PetscCall(MatSetSizes(A, 9, 9, 9, 9));
    PetscCall(MatSetType(A, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
    for (PetscInt i = 0; i < 9; ++i) {
        PetscCall(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY));

    Vec b;
    PetscCall(VecCreate(PETSC_COMM_SELF, &b));
    PetscCall(VecSetSizes(b, 9, 9));
    PetscCall(VecSetType(b, VECSEQ));
    PetscCall(VecSet(b, 1.0));

    // ---- Configure AMGX with the deliberately-invalid combination ----
    // -pc_amgx_amg_method AGGREGATION + default selector (PMIS) →
    // PCSetFromOptions_AMGX raises "Chosen selector is not used for AmgX
    // Aggregation AMG".
    PetscCall(PetscOptionsSetValue(NULL, "-pc_type",              "amgx"));
    PetscCall(PetscOptionsSetValue(NULL, "-pc_amgx_amg_method",   "AGGREGATION"));
    // intentionally NO -pc_amgx_selector

    KSP ksp;
    PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPGMRES));

    // KSPSetFromOptions is expected to fail. Capture the error code rather
    // than letting PetscCall propagate it (the whole point of this test is
    // that user code is permitted to catch and recover from such errors).
    std::printf("[test] calling KSPSetFromOptions (expected to fail)...\n");
    std::fflush(stdout);
    PetscErrorCode setopt_err = KSPSetFromOptions(ksp);
    if (!setopt_err) {
        std::printf("[test] WARNING: KSPSetFromOptions did NOT error. "
                    "Was the bug worked around at setopt time? "
                    "Test is inconclusive.\n");
        // Still test the destroy path.
    } else {
        std::printf("[test] KSPSetFromOptions errored as expected (rc=%d).\n",
                    (int)setopt_err);
    }
    std::fflush(stdout);

    // The load-bearing call: KSPDestroy on a partially-initialized KSP.
    // Without the fix, this aborts via MPI_Abort and the program never
    // returns to print the message below.
    std::printf("[test] calling KSPDestroy on the failed KSP...\n");
    std::fflush(stdout);
    PetscErrorCode destroy_err = KSPDestroy(&ksp);
    std::fflush(stdout);

    if (destroy_err) {
        std::printf("[FAIL] KSPDestroy errored (rc=%d) — destructor bug present\n",
                    (int)destroy_err);
        PetscCall(MatDestroy(&A));
        PetscCall(VecDestroy(&b));
        PetscCall(PetscFinalize());
        return 1;
    }

    std::printf("[PASS] KSPDestroy succeeded after PCSetFromOptions error\n");

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&b));
    PetscCall(PetscFinalize());
    return 0;
}
