#!/usr/bin/env python3
"""
diff_gmstab_dumps.py — pairwise compare gmstab dump CSVs from the C++
reference port and the PETSc port.

Usage:
    diff_gmstab_dumps.py <dump_dir>

Walks <dump_dir>/petsc_chkNN_<name>.csv vs <dump_dir>/cpp_chkNN_<name>.csv
and prints, per pair:
    chkNN <name>     shape: (m, n)    fro: ||A-B||_F   rel: ||A-B||_F / max(...)   max: max|A-B|     status
where status is "ok" if rel < 1e-10 AND max < 1e-10, "SIGN-FLIP" if
A == -B within tolerance, else "DRIFT".

Exits 0 if every pair passes; exits 1 on any drift; exits 2 on missing
file pairs or shape mismatches.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
import numpy as np


TOL = 1e-10


def load_csv(path: str) -> np.ndarray:
    """Load a CSV-of-doubles into a 2-D numpy array (m × n).
    Vector files (one value per line) load as (m, 1)."""
    rows: list[list[float]] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.split(",") if p]
            rows.append([float(p) for p in parts])
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    # If it's a column-vector dump (one element per row), all rows are length 1
    # and we return (m, 1).
    return np.array(rows, dtype=np.float64)


def cmp(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, str]:
    if a.shape != b.shape:
        return (np.inf, np.inf, np.inf, f"SHAPE-MISMATCH a={a.shape} b={b.shape}")
    diff = a - b
    fro_a = np.linalg.norm(a)
    fro_b = np.linalg.norm(b)
    fro_d = np.linalg.norm(diff)
    max_d = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(fro_a, fro_b, 1e-300)
    rel = fro_d / denom

    # Sign-flip check: is a == -b within tol?
    fro_sum = float(np.linalg.norm(a + b))
    rel_sum = fro_sum / denom

    if rel < TOL and max_d < TOL:
        status = "ok"
    elif rel_sum < TOL:
        status = "SIGN-FLIP"
    else:
        status = "DRIFT"
    return (float(fro_d), float(rel), max_d, status)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    dump_dir = argv[1]
    if not os.path.isdir(dump_dir):
        print(f"not a directory: {dump_dir}", file=sys.stderr)
        return 2

    # Enumerate (checkpoint, name) -> {port: path} mapping.
    pat = re.compile(r"^(petsc|cpp)_chk(\d+)_(.+)\.csv$")
    pairs: dict[tuple[int, str], dict[str, str]] = defaultdict(dict)
    for fname in sorted(os.listdir(dump_dir)):
        m = pat.match(fname)
        if not m:
            continue
        port, ckp, name = m.groups()
        key = (int(ckp), name)
        pairs[key][port] = os.path.join(dump_dir, fname)

    if not pairs:
        print(f"no gmstab dump files found under {dump_dir}", file=sys.stderr)
        return 2

    n_total = 0
    n_pass = 0
    n_drift = 0
    n_signflip = 0
    n_shape = 0
    n_missing = 0
    first_drift: tuple[int, str] | None = None

    print(f"{'chk':>3s} {'name':30s}  {'shape':>14s}  {'fro':>10s}  {'rel':>10s}  {'max':>10s}  status")
    print("-" * 110)

    for (ckp, name) in sorted(pairs.keys()):
        ports = pairs[(ckp, name)]
        if "petsc" not in ports or "cpp" not in ports:
            missing = "petsc" if "cpp" in ports else "cpp"
            print(f"{ckp:3d} {name:30s}  MISSING {missing} side")
            n_missing += 1
            n_total += 1
            continue
        try:
            a = load_csv(ports["petsc"])
            b = load_csv(ports["cpp"])
        except Exception as e:
            print(f"{ckp:3d} {name:30s}  LOAD-FAIL: {e}")
            n_total += 1
            continue
        fro, rel, mx, status = cmp(a, b)
        shape = f"{a.shape}" if a.shape == b.shape else f"{a.shape}/{b.shape}"
        print(f"{ckp:3d} {name:30s}  {shape:>14s}  {fro:10.3e}  {rel:10.3e}  {mx:10.3e}  {status}")
        n_total += 1
        if status == "ok":
            n_pass += 1
        elif status == "SIGN-FLIP":
            n_signflip += 1
            if first_drift is None:
                first_drift = (ckp, name)
        elif status == "DRIFT":
            n_drift += 1
            if first_drift is None:
                first_drift = (ckp, name)
        else:  # SHAPE-MISMATCH
            n_shape += 1
            if first_drift is None:
                first_drift = (ckp, name)

    print("-" * 110)
    print(f"summary: total={n_total} pass={n_pass} drift={n_drift} signflip={n_signflip} "
          f"shape-mismatch={n_shape} missing={n_missing}")
    if first_drift is not None:
        print(f"first divergence: chk{first_drift[0]:02d} {first_drift[1]}")
    return 0 if (n_drift == 0 and n_signflip == 0 and n_shape == 0 and n_missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
