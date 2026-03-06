"""
Diagnostic script — run FIRST after cloning AI4Risk/interbank.

Inspects the dataset folder structure, file formats, column names,
and prints everything needed to configure the loader.

Usage:
    git clone https://github.com/AI4Risk/interbank.git
    python inspect_data.py --path ./interbank/datasets

    # If datasets are elsewhere:
    python inspect_data.py --path /path/to/datasets/folder
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def inspect(data_path: str):
    root = Path(data_path)
    if not root.exists():
        print(f"ERROR: Path does not exist: {root}")
        print(f"\nDid you clone the repo?")
        print(f"  git clone https://github.com/AI4Risk/interbank.git")
        print(f"  python inspect_data.py --path ./interbank/datasets")
        sys.exit(1)

    print("=" * 70)
    print(f"  AI4Risk/interbank — Data Inspector")
    print(f"  Path: {root.resolve()}")
    print("=" * 70)

    # ── Step 1: List all files ────────────────────────────────────
    all_files = sorted(root.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    dirs_only = [f for f in all_files if f.is_dir()]

    print(f"\n  Total files: {len(files_only)}")
    print(f"  Total directories: {len(dirs_only)}")

    # Group by extension
    by_ext = defaultdict(list)
    for f in files_only:
        by_ext[f.suffix.lower()].append(f)

    print(f"\n  Files by extension:")
    for ext, files in sorted(by_ext.items()):
        print(f"    {ext or '(no ext)':>10s}: {len(files)} files")

    # ── Step 2: Show directory structure (first 2 levels) ────────
    print(f"\n  Directory structure:")
    for d in sorted(dirs_only):
        rel = d.relative_to(root)
        depth = len(rel.parts)
        if depth <= 2:
            indent = "    " + "  " * depth
            print(f"{indent}{d.name}/")

    # ── Step 3: List files matching quarter patterns ─────────────
    print(f"\n  Looking for quarterly data files...")
    quarter_files = [f for f in files_only if any(
        q in f.name for q in ["Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"]
    )]

    if quarter_files:
        print(f"  Found {len(quarter_files)} quarterly files:")
        # Show first 20
        for f in quarter_files[:20]:
            rel = f.relative_to(root)
            size_kb = f.stat().st_size / 1024
            print(f"    {str(rel):<50s} ({size_kb:.0f} KB)")
        if len(quarter_files) > 20:
            print(f"    ... and {len(quarter_files) - 20} more")
    else:
        print(f"  No quarterly files found. Looking for any CSV/NPY files...")
        for ext in [".csv", ".npy", ".npz", ".pkl", ".pt"]:
            if ext in by_ext:
                print(f"\n  {ext} files:")
                for f in by_ext[ext][:10]:
                    rel = f.relative_to(root)
                    print(f"    {str(rel)}")

    # ── Step 4: Inspect CSV/NPY file contents ────────────────────
    csv_files = by_ext.get(".csv", [])
    npy_files = by_ext.get(".npy", []) + by_ext.get(".npz", [])

    if csv_files:
        print(f"\n{'=' * 70}")
        print(f"  CSV File Inspection (first 3 files)")
        print(f"{'=' * 70}")
        _inspect_csvs(csv_files[:3])

    if npy_files:
        print(f"\n{'=' * 70}")
        print(f"  NPY/NPZ File Inspection (first 3 files)")
        print(f"{'=' * 70}")
        _inspect_npys(npy_files[:3])

    # Also look for feature / edge specific files
    feature_files = [f for f in files_only if "feature" in f.name.lower() or "feat" in f.name.lower()]
    edge_files = [f for f in files_only if "edge" in f.name.lower() or "adj" in f.name.lower()]
    target_files = [f for f in files_only if "target" in f.name.lower() or "label" in f.name.lower()]

    if feature_files:
        print(f"\n{'=' * 70}")
        print(f"  Feature files ({len(feature_files)} found)")
        print(f"{'=' * 70}")
        _inspect_csvs(feature_files[:2])
        _inspect_npys([f for f in feature_files if f.suffix in ['.npy', '.npz']][:2])

    if edge_files:
        print(f"\n{'=' * 70}")
        print(f"  Edge files ({len(edge_files)} found)")
        print(f"{'=' * 70}")
        _inspect_csvs([f for f in edge_files if f.suffix == '.csv'][:2])
        _inspect_npys([f for f in edge_files if f.suffix in ['.npy', '.npz']][:2])

    if target_files:
        print(f"\n{'=' * 70}")
        print(f"  Target/Label files ({len(target_files)} found)")
        print(f"{'=' * 70}")
        _inspect_csvs([f for f in target_files if f.suffix == '.csv'][:2])
        _inspect_npys([f for f in target_files if f.suffix in ['.npy', '.npz']][:2])

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — Copy this info when asking for loader help")
    print(f"{'=' * 70}")
    print(f"  Path: {root.resolve()}")
    print(f"  Total files: {len(files_only)}")
    print(f"  Extensions: {', '.join(sorted(by_ext.keys()))}")
    print(f"  Quarter files: {len(quarter_files)}")
    print(f"  Feature files: {len(feature_files)}")
    print(f"  Edge files: {len(edge_files)}")
    print(f"  Target files: {len(target_files)}")


def _inspect_csvs(files):
    """Print column names, shape, dtypes for CSV files."""
    try:
        import pandas as pd
    except ImportError:
        print("  (pandas not installed — skipping CSV inspection)")
        return

    for f in files:
        if f.suffix.lower() != '.csv':
            continue
        try:
            df = pd.read_csv(f, nrows=5)
            print(f"\n  File: {f.name}")
            print(f"    Shape (first 5 rows): {df.shape}")
            print(f"    Columns ({len(df.columns)}):")

            # Show first 10 and last 5 columns
            cols = df.columns.tolist()
            if len(cols) <= 15:
                for c in cols:
                    print(f"      - {c} ({df[c].dtype}): {df[c].iloc[0]}")
            else:
                print(f"    First 5:")
                for c in cols[:5]:
                    print(f"      - {c} ({df[c].dtype}): {df[c].iloc[0]}")
                print(f"    ... {len(cols) - 10} more ...")
                print(f"    Last 5:")
                for c in cols[-5:]:
                    print(f"      - {c} ({df[c].dtype}): {df[c].iloc[0]}")

            # Full row count
            full_df = pd.read_csv(f)
            print(f"    Total rows: {len(full_df)}")
        except Exception as e:
            print(f"\n  File: {f.name} — ERROR: {e}")


def _inspect_npys(files):
    """Print shape, dtype for NPY/NPZ files."""
    try:
        import numpy as np
    except ImportError:
        print("  (numpy not installed — skipping NPY inspection)")
        return

    for f in files:
        if f.suffix.lower() not in ['.npy', '.npz']:
            continue
        try:
            if f.suffix == '.npy':
                arr = np.load(f, allow_pickle=True)
                print(f"\n  File: {f.name}")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")
                print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]"
                      if arr.dtype.kind in 'fi' else f"    Sample: {arr.flat[:3]}")
            elif f.suffix == '.npz':
                data = np.load(f, allow_pickle=True)
                print(f"\n  File: {f.name}")
                print(f"    Keys: {list(data.keys())}")
                for k in list(data.keys())[:5]:
                    arr = data[k]
                    print(f"      {k}: shape={arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"\n  File: {f.name} — ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect AI4Risk data")
    parser.add_argument("--path", type=str, default="./interbank/datasets",
                        help="Path to the datasets folder")
    args = parser.parse_args()
    inspect(args.path)



