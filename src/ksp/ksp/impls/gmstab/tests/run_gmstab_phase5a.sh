#!/usr/bin/env bash
# Phase 5a — sweep all 129 baselines through PETSc gmstab and produce
# tests/results_phase5a/validation_summary_petsc.csv.
#
# Usage:
#   ./run_gmstab_phase5a.sh                 # run all 129
#   ./run_gmstab_phase5a.sh --filter X      # run only baselines whose name contains X
#                                           # (e.g., --filter cdr_sweep_small)
#   ./run_gmstab_phase5a.sh --skip-build    # don't rebuild libpetsc/harness

set -u

ROOT=/home/sam/hpc_stack/petsc
ARCH=arch-cuda-opt-i32
TESTS=$ROOT/src/ksp/ksp/impls/gmstab/tests
MPICC=/home/sam/.local/mpich/bin/mpicc

BASELINE_ROOT=/home/sam/hpc_stack/gmstab_matlab/gmstab_handoff_package_validation/baselines
RESULTS_DIR=$TESTS/results_phase5a

INCLUDES="-I$ROOT/include -I$ROOT/$ARCH/include"
LDFLAGS="-L$ROOT/$ARCH/lib -Wl,-rpath,$ROOT/$ARCH/lib -lpetsc -lm"
HARNESS_BIN=/tmp/ex_gmstab_phase5a_harness

# ---- args ----
FILTER=""
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --filter)     shift; FILTER="$1"; shift;;
    --skip-build) SKIP_BUILD=1; shift;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

# ---- build the harness ----
if [ "$SKIP_BUILD" = "0" ]; then
  echo "[5a] compiling harness..."
  $MPICC $INCLUDES "$TESTS/ex_gmstab_phase5a_harness.c" $LDFLAGS -o "$HARNESS_BIN" 2>&1 | grep -v "may conflict" || true
  if [ ! -x "$HARNESS_BIN" ]; then
    echo "[5a] harness compile FAILED"
    exit 2
  fi
fi

# ---- prepare results dir ----
mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="$RESULTS_DIR/validation_summary_petsc.csv"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

# CSV header
cat > "$SUMMARY_CSV" <<'EOF'
test,residual_tier,matvec_tier,overall,n_snap_P,n_snap_C,matvec_count_P,matvec_count_C,matvec_count_M,max_iterres_drift_PC,first_drift_iter,final_iterres_P,final_iterres_C,final_iterres_M,reason_P,cpp_converged,bnorm_actual,bnorm_summary,bnorm_warn,note
EOF

# ---- collect baseline directories ----
# Top-level: cdr_small, sherman5, ocean, asic_320ks, torso1
# Plus: cdr_sweep_small/* (125 sub-baselines)
BASELINES=()
for d in "$BASELINE_ROOT"/*/; do
  d="${d%/}"
  name="$(basename "$d")"
  # Skip the cdr_sweep_small parent itself; we descend into it below.
  if [ "$name" = "cdr_sweep_small" ]; then
    for sub in "$d"/*/; do
      sub="${sub%/}"
      BASELINES+=("$sub")
    done
  else
    # Only add if this looks like a baseline dir (has linsys.bin)
    if [ -f "$d/linsys.bin" ]; then
      BASELINES+=("$d")
    fi
  fi
done

# Apply filter
if [ -n "$FILTER" ]; then
  FILTERED=()
  for b in "${BASELINES[@]}"; do
    if [[ "$b" == *"$FILTER"* ]]; then
      FILTERED+=("$b")
    fi
  done
  BASELINES=("${FILTERED[@]}")
fi

total=${#BASELINES[@]}
echo "[5a] running $total baselines..."

# ---- run each baseline ----
pass=0
fail=0
err=0
warn=0
i=0
for b in "${BASELINES[@]}"; do
  i=$((i + 1))
  name="$(basename "$b")"
  log="$LOG_DIR/${name}.log"

  # Run the harness; capture stdout (the CSV row) and stderr (warnings) separately.
  out=$("$HARNESS_BIN" "$b" 2>"$log")
  rc=$?

  if [ $rc -eq 2 ]; then
    # Harness error — couldn't even run.
    echo "ERR  [$i/$total] $name (harness err) — see $log" >&2
    echo "$name,HARNESS_ERR,N/A,OVERALL_FAIL,-1,-1,-1,-1,-1,nan,-1,nan,nan,nan,-1,0,nan,nan,0,HARNESS_ERR" >> "$SUMMARY_CSV"
    err=$((err + 1))
    continue
  fi

  echo "$out" >> "$SUMMARY_CSV"

  # Quick parse: third comma-separated field is overall verdict.
  overall=$(echo "$out" | awk -F',' '{print $4}')
  if [ "$overall" = "OVERALL_PASS" ]; then
    pass=$((pass + 1))
    [ $((i % 20)) -eq 0 ] && echo "  [$i/$total] $name: PASS"
  else
    fail=$((fail + 1))
    echo "FAIL [$i/$total] $name — see $log" >&2
    # Show the harness summary line for debugging
    echo "    $out" >&2
  fi

  # Also count baselines with bnorm warnings
  bnorm_warn=$(echo "$out" | awk -F',' '{print $19}')
  if [ "$bnorm_warn" = "1" ]; then
    warn=$((warn + 1))
  fi
done

# ---- aggregate ----
echo ""
echo "[5a] ============================="
echo "[5a] $pass passed / $fail failed / $err harness-err  (of $total baselines)"
if [ $warn -gt 0 ]; then
  echo "[5a] $warn baselines emitted ||b|| sanity warnings (potential preconditioning leak)"
fi
echo "[5a] summary written to $SUMMARY_CSV"
echo "[5a] per-baseline logs in $LOG_DIR/"

# Compose aggregate.txt with key counts and verdict.
{
  echo "Phase 5a aggregate results"
  echo "Generated: $(date)"
  echo ""
  echo "Total baselines:    $total"
  echo "OVERALL_PASS:       $pass"
  echo "OVERALL_FAIL:       $fail"
  echo "HARNESS_ERR:        $err"
  echo "bnorm warnings:     $warn"
  echo ""
  if [ $fail -eq 0 ] && [ $err -eq 0 ]; then
    echo "Overall verdict: PHASE 5A PASS"
  else
    echo "Overall verdict: PHASE 5A NEEDS_INVESTIGATION"
  fi
} > "$RESULTS_DIR/aggregate.txt"

cat "$RESULTS_DIR/aggregate.txt"

# Exit code: 0 if all passed, non-zero otherwise
if [ $fail -gt 0 ] || [ $err -gt 0 ]; then
  exit 1
fi
exit 0
