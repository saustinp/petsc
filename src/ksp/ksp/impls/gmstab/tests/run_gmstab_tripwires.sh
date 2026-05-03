#!/usr/bin/env bash
# Phase-3 tripwire harness — rebuilds libpetsc, recompiles every gmstab
# validator, and runs them in order. Exits 0 only if all validators pass.
#
# Usage: ./run_gmstab_tripwires.sh                         # full rebuild + run
#        SKIP_BUILD=1 ./run_gmstab_tripwires.sh            # skip libpetsc rebuild
#
# This is the "make check" of Phase 3 — run it before committing any
# changes to src/ksp/ksp/impls/gmstab/.

set -u
ROOT=/home/sam/hpc_stack/petsc
ARCH=arch-cuda-opt-i32
TESTS=$ROOT/src/ksp/ksp/impls/gmstab/tests
MPICC=/home/sam/.local/mpich/bin/mpicc

INCLUDES="-I$ROOT/include -I$ROOT/$ARCH/include"
LDFLAGS="-L$ROOT/$ARCH/lib -Wl,-rpath,$ROOT/$ARCH/lib -lpetsc -lm"

VALIDATORS=(
  "ex_gmstab_phase3a"               # Initialisation bit-equivalence
  "ex_gmstab_init_nonzero_guess"    # constructor row stays correct with x0 != 0
  "ex_gmstab_cycle1"                # Cycle1 force_l1 trace vs C++ force_l1 trace
)

# Parallel runs of cycle1 (uses the same binary, varies mpiexec -n).
# Each rank count exercises a different MPI reduction order; cycle1
# must produce a residual matching the C++ sequential reference within
# the parallel tolerance (1e-7 in the validator).
PARALLEL_RANKS=(2 4 8)
MPIEXEC=/home/sam/.local/mpich/bin/mpiexec

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "[tripwires] rebuilding libpetsc..."
  ( cd "$ROOT" && PETSC_DIR=$ROOT PETSC_ARCH=$ARCH make libs ) || {
    echo "[tripwires] libpetsc rebuild FAILED"
    exit 2
  }
fi

fail_count=0
total=${#VALIDATORS[@]}
for v in "${VALIDATORS[@]}"; do
  src=$TESTS/$v.c
  bin=/tmp/$v
  if [ ! -f "$src" ]; then
    echo "[tripwires] $v: SKIPPED — source $src missing"
    continue
  fi
  echo "[tripwires] compiling $v..."
  if ! $MPICC $INCLUDES "$src" $LDFLAGS -o "$bin" 2>&1; then
    echo "[tripwires] $v: COMPILE FAILED"
    fail_count=$((fail_count + 1))
    continue
  fi
  echo "[tripwires] running $v..."
  if "$bin" >"/tmp/${v}.log" 2>&1; then
    echo "[tripwires] $v: PASS"
  else
    rc=$?
    echo "[tripwires] $v: FAIL (exit $rc) — see /tmp/${v}.log"
    tail -20 "/tmp/${v}.log" | sed 's/^/[tripwires]   /'
    fail_count=$((fail_count + 1))
  fi
done

echo ""
echo "[tripwires] sequential summary: $((total - fail_count))/$total passed"

# ---- Parallel cycle1 runs ----
parallel_total=${#PARALLEL_RANKS[@]}
parallel_fails=0
cycle1_bin=/tmp/ex_gmstab_cycle1
if [ -f "$cycle1_bin" ]; then
  for n in "${PARALLEL_RANKS[@]}"; do
    echo "[tripwires] running ex_gmstab_cycle1 on $n MPI ranks..."
    if "$MPIEXEC" -n "$n" "$cycle1_bin" >"/tmp/ex_gmstab_cycle1.np${n}.log" 2>&1; then
      echo "[tripwires] ex_gmstab_cycle1 (n=$n): PASS"
    else
      rc=$?
      echo "[tripwires] ex_gmstab_cycle1 (n=$n): FAIL (exit $rc) — see /tmp/ex_gmstab_cycle1.np${n}.log"
      tail -10 "/tmp/ex_gmstab_cycle1.np${n}.log" | sed 's/^/[tripwires]   /'
      parallel_fails=$((parallel_fails + 1))
    fi
  done
else
  echo "[tripwires] $cycle1_bin missing — skipping parallel runs"
  parallel_total=0
fi

echo ""
echo "[tripwires] parallel summary: $((parallel_total - parallel_fails))/$parallel_total passed (ranks: ${PARALLEL_RANKS[*]})"
echo "[tripwires] OVERALL: $((total + parallel_total - fail_count - parallel_fails))/$((total + parallel_total)) passed"
exit $((fail_count + parallel_fails))
