#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text.py — Generate quick dataset summary and a lightweight markdown report for Lifeline (CTG).
- Loads UCI CTG via ucimlrepo (id=193) or a local CSV fallback.
- Prints class distribution and feature list.
- Optionally writes a README-like markdown report to disk.
Usage:
    python text.py --out report.md
    python text.py --csv ./data/CTG.csv --out report.md
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

def load_data(csv_path: str | None):
    X, y = None, None
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'NSP' not in df.columns:
            raise ValueError("CSV must contain target column 'NSP'.")
        y = df['NSP']
        # Feature columns used in your notebook
        feature_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                        'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                        'Median', 'Variance', 'Tendency']
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        X = df[feature_cols]
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except Exception as e:
            raise RuntimeError("ucimlrepo not installed and no CSV provided. "
                               "Install ucimlrepo or pass --csv <path>.") from e
        ctg = fetch_ucirepo(id=193)
        X = ctg.data.features
        y = ctg.data.targets['NSP']
    return X, y

def summarize(X: pd.DataFrame, y: pd.Series) -> dict:
    counts = {
        'normal': int((y == 1).sum() if y.min() == 1 else (y == 0).sum()),
        'suspect': int((y == 2).sum() if y.max() >= 2 else (y == 0.5).sum()),
        'pathological': int((y == 3).sum() if y.max() >= 3 else (y == 1).sum())
    }
    info = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'class_counts': counts
    }
    return info

def make_markdown(info: dict) -> str:
    lines = []
    lines.append("# Lifeline – CTG Dataset Summary\n")
    lines.append(f"- **Samples**: {info['n_samples']}")
    lines.append(f"- **Features**: {info['n_features']}")
    cc = info['class_counts']
    lines.append(f"- **Class distribution**: Normal={cc['normal']}, Suspect={cc['suspect']}, Pathological={cc['pathological']}")
    lines.append("\n## Features\n")
    lines.append(", ".join(info['feature_names']))
    lines.append("\n## Notes\n")
    lines.append("- Suspect often overlaps Normal/Pathological. Consider calibrated thresholds or margin bands.\n")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional path to CTG.csv")
    ap.add_argument("--out", type=str, default=None, help="Optional markdown output file")
    args = ap.parse_args()

    X, y = load_data(args.csv)
    info = summarize(X, y)
    md = make_markdown(info)

    print(md)
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n[✓] Wrote summary to {out_path}")

if __name__ == "__main__":
    sys.exit(main())
