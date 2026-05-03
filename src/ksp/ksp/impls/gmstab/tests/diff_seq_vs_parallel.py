#!/usr/bin/env python3
"""
diff_seq_vs_parallel.py — verify a cycle's intermediate dumps match
between sequential and parallel runs of the same PETSc validator.

Both runs go through the same code path (no gauge difference between
LAPACK calls within a single PETSc port — both seq and parallel call
the same dgesdd / dgeqrf / etc.), so EVERY checkpoint should match at
machine precision modulo MPI Allreduce reordering rounding (~1e-12).

Unlike diff_gmstab_dumps.py (which compares PETSc to the C++ port and
must ignore gauge-dependent intermediate stages), this tool compares
PETSc to PETSc and asserts ALL quantities match. Any drift exceeding
1e-10 is a real bug — most likely an MPI-vs-sequential code-path
divergence that has no place in a deterministic algorithm.

Usage:
    diff_seq_vs_parallel.py <seq_dump_dir> <par_dump_dir> [--skip name1[,name2[,...]]]

The optional --skip flag whitelists checkpoint quantities that are
documented as exempt from the bit-match requirement (e.g., cycle 1's
V0_postBGS, which is the rank-deficiency-noise unit-vector — see
PHASE3_STATUS.md). The skipped quantities are still printed but
don't contribute to the pass/fail gate.

Returns 0 on PASS, 1 on any unwhitelisted drift.
"""
import os
import sys
import numpy as np

TOL = 1e-10


def load(p):
    return np.atleast_2d(np.genfromtxt(p, delimiter=','))


def parse_args(argv):
    if len(argv) < 3:
        return None, None, None
    seq_dir, par_dir = argv[1], argv[2]
    skip = set()
    if len(argv) >= 5 and argv[3] == "--skip":
        skip = {s.strip() for s in argv[4].split(",") if s.strip()}
    return seq_dir, par_dir, skip


def main(argv):
    seq_dir, par_dir, skip = parse_args(argv)
    if seq_dir is None:
        print(__doc__, file=sys.stderr)
        return 2
    files = sorted(f for f in os.listdir(seq_dir) if f.startswith("petsc_"))
    n_total = 0
    n_drift = 0
    n_skip_drift = 0
    worst = ("", 0.0)
    print(f"{'file':45s}  {'rel':>10s}  {'max':>10s}  status")
    print("-" * 90)
    for f in files:
        a_path = os.path.join(seq_dir, f)
        b_path = os.path.join(par_dir, f)
        # Strip "petsc_" prefix and ".csv" suffix to get the checkpoint name.
        qname = f.removeprefix("petsc_").removesuffix(".csv")
        is_skipped = qname in skip
        if not os.path.exists(b_path):
            print(f"{f:45s}  MISSING in {par_dir}")
            if not is_skipped:
                n_drift += 1
            continue
        a = load(a_path)
        b = load(b_path)
        if a.shape != b.shape:
            print(f"{f:45s}  SHAPE MISMATCH a={a.shape} b={b.shape}")
            if not is_skipped:
                n_drift += 1
            continue
        diff = a - b
        fro = float(np.linalg.norm(diff))
        max_ = float(np.max(np.abs(diff))) if diff.size else 0.0
        denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)
        rel = fro / denom
        if (not is_skipped) and rel > worst[1]:
            worst = (f, rel)
        if rel < TOL:
            status = "ok"
        elif is_skipped:
            status = "SKIP-DRIFT (whitelisted)"
            n_skip_drift += 1
        else:
            status = "DRIFT"
            n_drift += 1
        print(f"{f:45s}  {rel:10.3e}  {max_:10.3e}  {status}")
        n_total += 1
    print("-" * 90)
    print(f"summary: total={n_total}  drift={n_drift}  skip-drift={n_skip_drift}  "
          f"worst={worst[1]:.3e} on {worst[0]}")
    return 0 if n_drift == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
