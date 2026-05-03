#!/usr/bin/env python3
"""
diff_natural_seq_vs_par.py — verify the natural-flow trace from a
sequential PETSc run agrees with a parallel PETSc run of the same code,
operator, and shadow space.

Both runs use LAPACK on identical input data. The only source of
divergence is MPI Allreduce reordering, which is bounded by ~1e-12 per
reduction. The Init prefix (rows 0..INIT_PREFIX-1) is produced inside the
inner gmres_m and uses no SVD/QR/LQ — drift here should be at machine
precision modulo a small MPI-reorder constant. The matvec column should
match for the entire prefix.

Outside the Init prefix, drift accumulates as in the C++ comparison
(the polynomial step amplifies even Allreduce noise by 50-200× per
cycle). We don't gate on per-row drift past row INIT_PREFIX-1; we only
check that:

    - Init prefix (rows 0..INIT_PREFIX-1) match at <= TIGHT_TOL drift
      on iter and true residual columns.
    - matvec column matches exactly for the Init prefix.
    - Both runs reach a similar number of total snapshots (within 25%).

Usage:
    diff_natural_seq_vs_par.py <seq.csv> <par.csv>

Exit 0 on PASS, 1 on FAIL.
"""
import sys

INIT_PREFIX = 12
TIGHT_TOL   = 1e-10        # Allreduce-only drift on gauge-independent rows
ROW_FUDGE   = 1.25         # parallel snapshot count within 25% of seq


def parse_csv(path):
    rows = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            iter_idx = int(parts[0])
            mv       = int(parts[1])
            iterres  = float(parts[2])
            trueres  = float(parts[3])
            rows.append((iter_idx, mv, iterres, trueres))
    return rows


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    seq = parse_csv(argv[1])
    par = parse_csv(argv[2])
    n_seq, n_par = len(seq), len(par)

    print(f"seq rows = {n_seq}, par rows = {n_par}")

    fail = 0
    prefix = min(INIT_PREFIX, n_seq, n_par)
    if prefix < INIT_PREFIX:
        print(f"FAIL: trace shorter than INIT_PREFIX={INIT_PREFIX}: seq={n_seq} par={n_par}")
        fail = 1

    worst_iter = 0.0
    worst_true = 0.0
    mv_match = True
    for i in range(prefix):
        s_i, s_m, s_ir, s_tr = seq[i]
        p_i, p_m, p_ir, p_tr = par[i]
        if s_m != p_m:
            print(f"row {i}: matvec mismatch seq={s_m} par={p_m}")
            mv_match = False
        d_iter = abs(s_ir - p_ir)
        d_true = abs(s_tr - p_tr)
        if d_iter > worst_iter: worst_iter = d_iter
        if d_true > worst_true: worst_true = d_true
        status = "ok" if (s_m == p_m and d_iter < TIGHT_TOL and d_true < TIGHT_TOL) else "DRIFT"
        print(f"row {i:3d}: seq=(mv={s_m}, iter={s_ir:.6e})  par=(mv={p_m}, iter={p_ir:.6e})  d_iter={d_iter:.3e}  d_true={d_true:.3e}  {status}")

    print(f"Init prefix worst drift: iter={worst_iter:.3e}  true={worst_true:.3e}")
    g_drift = (worst_iter < TIGHT_TOL and worst_true < TIGHT_TOL)
    g_mv    = mv_match
    g_count = (n_par <= ROW_FUDGE * n_seq) and (n_seq <= ROW_FUDGE * n_par)

    print(f"  gate Init prefix drift  : worst={max(worst_iter,worst_true):.3e} (tol {TIGHT_TOL:.0e}) -> {'PASS' if g_drift else 'FAIL'}")
    print(f"  gate Init prefix matvec : -> {'PASS' if g_mv else 'FAIL'}")
    print(f"  gate row count parity   : seq={n_seq} par={n_par} (within {ROW_FUDGE}x) -> {'PASS' if g_count else 'FAIL'}")

    if not (g_drift and g_mv and g_count):
        fail = 1
    print("PASS" if fail == 0 else "FAIL")
    return fail


if __name__ == "__main__":
    sys.exit(main(sys.argv))
