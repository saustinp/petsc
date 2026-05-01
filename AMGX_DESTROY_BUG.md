# Bug report — `PCDestroy_AMGX` fails on partially-initialized PC

**Component:** `src/ksp/pc/impls/amgx/amgx.cxx`
**Severity:** Medium — affects sweep / batch tooling that catches and recovers from
configuration errors. A single mis-configured AMGX preconditioner currently cannot
be recovered from; the second-error chain calls `MPI_Abort` and kills the process.
**PETSc version:** Development branch, `v3.25.0-171-g7d1cbee15b2` (2026-04-30).
**AMGX version:** 2.4.0 (per `--download-amgx`).
**Configure options:** `--with-cuda=1 --download-amgx --download-hypre ...` (full
text in the stack trace below).

---

## Summary

`PCDestroy_AMGX` (`amgx.cxx:403`) enforces `amgx->rsrc != nullptr` via a
`PetscCheck`, but `amgx->rsrc` can legitimately be null when the destructor is
called on a PC whose `PCSetFromOptions_AMGX` errored *before* allocating
resources. PETSc's "second error after first error" handler then triggers
`MPI_Abort`, killing the entire process. This makes it impossible for a sweep /
batch driver to gracefully skip a misconfigured AMGX configuration and continue
with the next one.

A representative failure mode: the AMGX binding initializes `amgx->selector` to
a value that is valid for CLASSICAL but not for AGGREGATION. If the user
selects `AGGREGATION` without explicitly overriding the selector,
`PCSetFromOptions_AMGX` correctly errors at `amgx.cxx:487`. But the subsequent
`KSPDestroy(&ksp)` aborts the process via the destructor bug.

---

## Impact scope — when does this block users?

This is a **recovery-path bug**, not a "can't use AMGX" bug. The matrix below
clarifies what kinds of usage are affected.

| use case | works without the destructor fix? |
|---|---|
| Single application using AMGX with correct options on the first try | ✅ **Yes — fully usable.** Validation passes, `rsrc` is allocated, the destructor exits cleanly because `rsrc` is non-null. |
| Sweep / batch tool that tries many AMGX configs and needs to recover from errors per-config | ⚠️ **Blocked.** Any single misconfigured AMGX entry aborts the entire sweep via `MPI_Abort` at the destructor. |
| Application that catches a `PCSetFromOptions` error and falls back to a different PC | ⚠️ **Blocked.** The fallback path's `KSPDestroy` triggers the second-error abort before control reaches the user's fallback handler. |
| Application that catches `PCSetFromOptions` error, then exits cleanly without `KSPDestroy` (i.e. relies on process exit to free resources) | ✅ Works, but leaks. Not a real workaround for long-running apps. |

The production users most affected are **automated tooling**: parameter
sweeps, CI test matrices, autotuning harnesses, and applications that gracefully
swap AMG implementations at runtime. Day-to-day single-config users are not
affected as long as they pass the correct options.

For maintainer triage: severity is "blocking for tooling, invisible for normal
single-config use." A simple, low-risk patch (defensive null guards in the
destructor) restores graceful recovery for the affected class of users.

---

## Reproducer

Minimal triggering invocation (any matrix; the failure is in option parsing,
not the solve):

```bash
mpirun -n 1 ./your_app \
    -mat_type aijcusparse -vec_type cuda \
    -ksp_type gmres \
    -pc_type amgx \
    -pc_amgx_amg_method AGGREGATION
    # Note: no -pc_amgx_selector, so default selector (PMIS-class) is invalid
    # for AGGREGATION method.
```

Expected: `PCSetFromOptions_AMGX` raises an actionable error explaining that
the user needs `-pc_amgx_selector {SIZE_2,SIZE_4,SIZE_8,MULTI_PAIRWISE}` for
AGGREGATION. Application catches the error, destroys the KSP, continues.

Actual: First error reaches the user code via `PCSetFromOptions` →
`KSPSetFromOptions` return chain (good). Application calls
`KSPDestroy(&ksp)`. Inside, `PCDestroy_AMGX` runs the
`PetscCheck(amgx->rsrc != nullptr, ..., "s_rsrc == NULL")` line, which fires
because `rsrc` was never allocated. PETSc detects two errors in flight,
prints the "previous unhandled error / next error" warning, and calls
`MPI_Abort(MPI_COMM_SELF, 77)`. Process terminates. (As a side effect, the
AMGX library's atexit destructors run during shutdown and produce
`malloc(): unsorted double linked list corrupted`.)

---

## Stack trace (excerpt)

First error (correct, expected, recoverable):

```
[0]PETSC ERROR: PETSc has generated inconsistent data
[0]PETSC ERROR: Chosen selector is not used for AmgX Aggregation AMG
[0]PETSC ERROR: #1 PCSetFromOptions_AMGX() at .../amgx.cxx:487
[0]PETSC ERROR: #2 PCSetFromOptions() at .../pcset.c:151
[0]PETSC ERROR: #3 KSPSetFromOptions() at .../itcl.c:647
```

Second error (the bug — destructor on partial init):

```
[0]PETSC ERROR: It appears a new error in the code was triggered after a
                previous error, possibly because:
[0]PETSC ERROR:   - The first error was not properly handled ...
[0]PETSC ERROR:   - The second error was triggered while handling the first
                  error.
[0]PETSC ERROR: PETSc has generated inconsistent data
[0]PETSC ERROR: s_rsrc == NULL
[0]PETSC ERROR: #1 PCDestroy_AMGX() at .../amgx.cxx:403
[0]PETSC ERROR: #2 PCDestroy() at .../precon.c:146
[0]PETSC ERROR: #3 KSPDestroy() at .../itfunc.c:1537
```

`MPI_Abort(MPI_COMM_SELF, 77)` follows immediately.

---

## Root cause

`PCDestroy_AMGX` (lines 395–416 of `amgx.cxx`):

```cpp
static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (s_count == 1) {
    /* can put this in a PCAMGXInitializePackage method */
    PetscCheck(amgx->rsrc != nullptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "s_rsrc == NULL");
    PetscCallAmgX(AMGX_resources_destroy(amgx->rsrc));
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
    PetscCallAmgX(AMGX_finalize_plugins());
    PetscCallAmgX(AMGX_finalize());
    PetscCallMPI(MPI_Comm_free(&amgx->comm));
  } else {
    PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
  }
  s_count -= 1;
  PetscCall(PetscFree(amgx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

Two issues:

1. **`amgx->rsrc != nullptr` enforced as an error rather than as a guard.** When
   the constructor / `PCSetFromOptions_AMGX` errored before allocating `rsrc`,
   the destructor cannot proceed cleanly *and* cannot opt out of the resource
   destruction calls. The result: every cleanup path errors.

2. **Other handles in the destructor (`amgx->cfg`, `amgx->comm`) are not
   guarded for null either.** Similar partial-init scenarios could fail at
   `AMGX_config_destroy(amgx->cfg)` or `MPI_Comm_free(&amgx->comm)`.

---

## Suggested fix

Make the destructor defensive about null pointers. Each cleanup step should be
gated on the corresponding handle being non-null. Approximately:

```cpp
static PetscErrorCode PCDestroy_AMGX(PC pc)
{
  PC_AMGX *amgx = (PC_AMGX *)pc->data;

  PetscFunctionBegin;
  if (s_count == 1) {
    if (amgx->rsrc) {
      PetscCallAmgX(AMGX_resources_destroy(amgx->rsrc));
      amgx->rsrc = nullptr;
    }
    if (amgx->cfg) {
      PetscCallAmgX(AMGX_config_destroy(amgx->cfg));
      amgx->cfg = nullptr;
    }
    /* AMGX_finalize_plugins / AMGX_finalize are global, so only call
       if AMGX was actually initialized. Guard with a static bool or
       check s_count semantics. */
    PetscCallAmgX(AMGX_finalize_plugins());
    PetscCallAmgX(AMGX_finalize());
    if (amgx->comm != MPI_COMM_NULL) {
      PetscCallMPI(MPI_Comm_free(&amgx->comm));
    }
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
```

Also worth verifying that `PCCreate_AMGX` initializes all of `rsrc`, `cfg`,
`comm`, and `solver` to null/invalid sentinel values, so the null guards
behave deterministically when the constructor partial-fails.

A regression test: a `PCSetFromOptions` call that fails (e.g. unsupported
selector + AGGREGATION method) should be followed by `PCDestroy()` returning
cleanly without raising a second error.

---

## Workarounds (currently in use)

In our sweep driver, we now call `KSPDestroy` without `PetscCall` and discard
its return code (`(void)KSPDestroy(&ksp);`) for any KSP whose
`PCSetFromOptions` errored. This avoids the second-error → MPI_Abort cascade
at the cost of a small (few-KB) leak per misconfigured PC. With this
workaround in place, an AMGX-using sweep can survive misconfigured AMGX
entries — but only if the *first* AMGX entry to be torn down isn't the one
that holds the package's global init state.

This workaround is enough to put AMGX back in production sweeps as long as
configurations are vetted ahead of time (so misconfig is the exception, not
the rule). For applications that catch errors and dynamically swap PCs at
runtime, the workaround is fragile because user code does not control the
order of `KSPDestroy` calls relative to the `s_count` package state.

---

## Effort estimate to fix upstream

- **Code change:** ~10 lines in `amgx.cxx`. Should take 30 min to write.
- **Verification:** ~2 hours, including a small regression test that
  exercises the `PCSetFromOptions failure → PCDestroy must succeed` path.
- **Review:** standard PETSc workflow.

The diagnostics for "missing AGGREGATION selector" itself could optionally be
improved as well — the current error message says "Chosen selector is not
used for AmgX Aggregation AMG" without naming which selectors *are* valid.
A friendlier message would help users avoid the partial-init path in the
first place. Suggested:

```
"For AmgX Aggregation AMG, -pc_amgx_selector must be one of "
"{SIZE_2, SIZE_4, SIZE_8, MULTI_PAIRWISE}; got '%s'"
```

---

## Submitter notes

- Discovered while building a comparison harness that runs PETSc's AMG
  preconditioner sweep across a set of test matrices. The sweep driver was
  designed to catch per-config errors (out-of-memory, unsupported option,
  numerical breakdown) and continue with the next config.
- The `PCDestroy_AMGX` failure is the only one in the sweep that escalates
  to `MPI_Abort` rather than letting the driver record the trial as
  errored and continue.
- The selector-validation error is itself fine and useful — only the
  destructor's behavior on the post-error cleanup path is the bug.
- Reproducer and full stack trace can be made available if helpful;
  contact via the channel of submission.
