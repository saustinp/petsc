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
  "ex_gmstab_cycle2"                # Cycle2 force_l2 trace vs C++ force_l2 trace
  "ex_gmstab_natural"               # full natural-flow flying-restart driver
  "ex_gmstab_natural_nzg"           # natural-flow with KSPSetInitialGuessNonzero (x_global accumulation)
  "ex_gmstab_determinism"           # same-rank repeatability (byte-identical traces across two runs)
  "ex_gmstab_multisolve"            # KSPSolve called twice on same KSP — bit-identical x's
  "ex_gmstab_multisolve_rng"        # multi-solve via default-RNG shadow path (re-seed on every call)
  "ex_gmstab_schange"               # s changes between two KSPSolve calls (V0/V1/Z re-alloc at correct size)
  "ex_gmstab_pcright_jacobi"        # Phase 4a: PC_RIGHT + Jacobi (zero initial guess)
  "ex_gmstab_pcright_nzg"           # Phase 4a: PC_RIGHT + Jacobi + nonzero initial guess (x_initial save/restore)
  "ex_gmstab_pcleft_jacobi"         # Phase 4b: PC_LEFT + Jacobi
  "ex_gmstab_pcleft_bjacobi"        # Phase 4b: PC_LEFT + BJacobi (skips n=8 stall)
  "ex_gmstab_pc_sweep"              # Phase 4d: cross-product (pc_side, pc_type) sweep on cdr_small
)

# Parallel runs (same binaries, varies mpiexec -n).
# Each rank count exercises a different MPI reduction order; the
# validators must produce a residual matching the C++ sequential reference
# within the parallel tolerance (1e-7 for cycle1/cycle2 init prefix; same
# for natural-flow which gates only on the gauge-independent init prefix
# at machine precision and on structural correctness elsewhere).
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
parallel_total=0
parallel_fails=0
PARALLEL_VALIDATORS=(ex_gmstab_cycle1 ex_gmstab_cycle2 ex_gmstab_natural
                     ex_gmstab_natural_nzg ex_gmstab_determinism ex_gmstab_multisolve
                     ex_gmstab_multisolve_rng ex_gmstab_schange
                     ex_gmstab_pcright_jacobi ex_gmstab_pcright_nzg
                     ex_gmstab_pcleft_jacobi ex_gmstab_pcleft_bjacobi
                     ex_gmstab_pc_sweep)
for v in "${PARALLEL_VALIDATORS[@]}"; do
  bin=/tmp/$v
  if [ ! -f "$bin" ]; then
    echo "[tripwires] $bin missing — skipping its parallel runs"
    continue
  fi
  for n in "${PARALLEL_RANKS[@]}"; do
    parallel_total=$((parallel_total + 1))
    echo "[tripwires] running $v on $n MPI ranks..."
    if "$MPIEXEC" -n "$n" "$bin" >"/tmp/${v}.np${n}.log" 2>&1; then
      echo "[tripwires] $v (n=$n): PASS"
    else
      rc=$?
      echo "[tripwires] $v (n=$n): FAIL (exit $rc) — see /tmp/${v}.np${n}.log"
      tail -10 "/tmp/${v}.np${n}.log" | sed 's/^/[tripwires]   /'
      parallel_fails=$((parallel_fails + 1))
    fi
  done
done

echo ""
echo "[tripwires] parallel summary: $((parallel_total - parallel_fails))/$parallel_total passed (ranks: ${PARALLEL_RANKS[*]})"

# ---- Sequential vs parallel intermediate-dump cross-check ----
# This catches bugs where seq and parallel code paths drift apart at
# the per-checkpoint level even when the per-snapshot residual trace
# happens to agree (e.g., a cancellation that masks an intermediate
# divergence). Both seq and parallel PETSc use the same LAPACK calls
# (so no gauge difference between runs), so EVERY checkpoint should
# match modulo MPI Allreduce reordering rounding.
xfails=0
xtotal=0
DIFF_TOOL="$TESTS/diff_seq_vs_parallel.py"
SEQ_PAR_VALIDATORS=(ex_gmstab_cycle1 ex_gmstab_cycle2)
SEQ_PAR_RANKS=(2 4)
# Per-validator skip lists: V0_postBGS in cycle 1 is the documented
# rank-deficiency-noise unit-vector that won't bit-match across rank
# counts (see PHASE3_STATUS.md "V0_postBGS rank-deficiency noise").
declare -A SKIP_FOR
SKIP_FOR[ex_gmstab_cycle1]="chk04_V0_postBGS"
SKIP_FOR[ex_gmstab_cycle2]=""
for v in "${SEQ_PAR_VALIDATORS[@]}"; do
  bin=/tmp/$v
  if [ ! -f "$bin" ]; then continue; fi
  # Run seq dump.
  seq_dir=/tmp/${v}_seq_dump
  rm -rf "$seq_dir"; mkdir -p "$seq_dir"
  GMSTAB_DUMP_DIR="$seq_dir" "$bin" >/dev/null 2>&1 || true
  skip_arg=""
  if [ -n "${SKIP_FOR[$v]:-}" ]; then
    skip_arg="--skip ${SKIP_FOR[$v]}"
  fi
  for n in "${SEQ_PAR_RANKS[@]}"; do
    par_dir=/tmp/${v}_p${n}_dump
    rm -rf "$par_dir"; mkdir -p "$par_dir"
    xtotal=$((xtotal + 1))
    GMSTAB_DUMP_DIR="$par_dir" "$MPIEXEC" -n "$n" "$bin" >/dev/null 2>&1 || true
    echo "[tripwires] seq-vs-${n}rank dump diff for $v..."
    if python3 "$DIFF_TOOL" "$seq_dir" "$par_dir" $skip_arg >"/tmp/${v}_seq_p${n}.diff.log" 2>&1; then
      worst=$(tail -1 "/tmp/${v}_seq_p${n}.diff.log" | grep -oP 'worst=\S+')
      echo "[tripwires] seq-vs-${n}rank ($v): PASS  ($worst)"
    else
      echo "[tripwires] seq-vs-${n}rank ($v): FAIL — see /tmp/${v}_seq_p${n}.diff.log"
      tail -10 "/tmp/${v}_seq_p${n}.diff.log" | sed 's/^/[tripwires]   /'
      xfails=$((xfails + 1))
    fi
  done
done

echo ""
echo "[tripwires] seq-vs-parallel cross-check: $((xtotal - xfails))/$xtotal passed"

# ---- Natural-flow seq-vs-parallel residual-trace coherence ----
# Both runs of ex_gmstab_natural use LAPACK on identical input data.
# The Init prefix (rows 0..11) is gauge-INDEPENDENT — produced by
# inner gmres_m with no SVD/QR/LQ. Drift between seq and parallel here
# is purely MPI Allreduce reordering, which should be ≤ 1e-12. The
# matvec column must match exactly through that prefix. Beyond row 11,
# the polynomial step amplifies even Allreduce noise so we don't gate
# on per-row drift.
nfails=0
ntotal=0
NATURAL_DIFF_TOOL="$TESTS/diff_natural_seq_vs_par.py"
NATURAL_BIN=/tmp/ex_gmstab_natural
NATURAL_TRACE=/tmp/petsc_natural_residuals.csv
if [ -f "$NATURAL_BIN" ]; then
  # Re-run sequential to repopulate the trace (the parallel runs above
  # may have left a parallel-mode trace at $NATURAL_TRACE).
  "$NATURAL_BIN" >/dev/null 2>&1 || true
  cp "$NATURAL_TRACE" /tmp/petsc_natural_residuals.seq.csv 2>/dev/null || true

  for n in 2 4; do
    ntotal=$((ntotal + 1))
    "$MPIEXEC" -n "$n" "$NATURAL_BIN" >/dev/null 2>&1 || true
    cp "$NATURAL_TRACE" /tmp/petsc_natural_residuals.p${n}.csv 2>/dev/null || true
    echo "[tripwires] natural seq-vs-${n}rank trace coherence..."
    if python3 "$NATURAL_DIFF_TOOL" \
         /tmp/petsc_natural_residuals.seq.csv \
         /tmp/petsc_natural_residuals.p${n}.csv \
         >"/tmp/natural_seq_p${n}.diff.log" 2>&1; then
      worst=$(grep -oP 'worst drift: iter=\S+' /tmp/natural_seq_p${n}.diff.log | head -1 || echo "")
      echo "[tripwires] natural seq-vs-${n}rank: PASS  ($worst)"
    else
      echo "[tripwires] natural seq-vs-${n}rank: FAIL — see /tmp/natural_seq_p${n}.diff.log"
      tail -15 /tmp/natural_seq_p${n}.diff.log | sed 's/^/[tripwires]   /'
      nfails=$((nfails + 1))
    fi
  done
fi
echo "[tripwires] natural seq-vs-parallel coherence: $((ntotal - nfails))/$ntotal passed"

echo "[tripwires] OVERALL: $((total + parallel_total + xtotal + ntotal - fail_count - parallel_fails - xfails - nfails))/$((total + parallel_total + xtotal + ntotal)) passed"
exit $((fail_count + parallel_fails + xfails + nfails))
