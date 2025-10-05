#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — End-to-end training/evaluation for Lifeline (CTG) classification.
Implements multiple models with stratified split and optional 5-fold CV.
Usage (examples):
    # Basic run with LightGBM
    python train.py --model lgbm

    # XGBoost with 5-fold CV and save model
    python train.py --model xgb --cv 5 --save_model xgb_model.pkl

    # Logistic regression (L1) with standard scaling
    python train.py --model logreg

    # RandomForest baseline
    python train.py --model rf

    # Use a local CSV instead of ucimlrepo
    python train.py --csv ./data/CTG.csv --model lgbm
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Optional imports guarded
def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    except Exception:
        return None

def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception:
        return None

def _try_import_imbens():
    try:
        from imbens.ensemble import SelfPacedEnsembleClassifier, SMOTEBoostClassifier
        return SelfPacedEnsembleClassifier, SMOTEBoostClassifier
    except Exception:
        return None, None

def load_data(csv_path: str | None):
    if csv_path:
        df = pd.read_csv(csv_path)
        y = df['NSP'].astype(int)
        feature_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                        'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                        'Median', 'Variance', 'Tendency']
        X = df[feature_cols].copy()
        return X, y
    else:
        from ucimlrepo import fetch_ucirepo
        ctg = fetch_ucirepo(id=193)
        X = ctg.data.features.copy()
        y = ctg.data.targets['NSP'].astype(int).copy()
        return X, y

def make_splits(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def get_model(name: str, seed: int = 43):
    name = name.lower()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial",
                                      max_iter=5000, C=0.1, random_state=seed))
        ])
        return model
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=seed))
        ])
        return model
    if name == "lgbm":
        LGBMClassifier = _try_import_lightgbm()
        if LGBMClassifier is None:
            # Fallback to RandomForest with a clear console notice
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            print("[!] lightgbm not installed — falling back to RandomForestClassifier. "
                  "To use LightGBM, run: pip install lightgbm")
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                               n_jobs=-1, random_state=seed))
            ])
            return model
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=-1,
                               random_state=seed, class_weight="balanced")
        return model
    if name == "xgb":
        XGBClassifier = _try_import_xgboost()
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=20,
                              subsample=0.8, colsample_bytree=0.8, random_state=seed,
                              use_label_encoder=False, eval_metric="mlogloss")
        return model
    if name == "spe":
        SPE, _ = _try_import_imbens()
        if SPE is None:
            raise RuntimeError("imbalanced-ensemble not installed. pip install imbalanced-ensemble")
        return SPE(random_state=seed)
    if name == "smoteboost":
        _, SB = _try_import_imbens()
        if SB is None:
            raise RuntimeError("imbalanced-ensemble not installed. pip install imbalanced-ensemble")
        return SB(random_state=seed)
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, metric="minkowski", weights="uniform"))
        ])
        return model
    raise ValueError(f"Unknown model '{name}'. Choose from [logreg, rf, lgbm, xgb, spe, smoteboost, knn].")

def encode_targets(y: pd.Series) -> pd.Series:
    # Map to 0,1,2 to match scikit-learn expectations
    return y.map({1:0, 2:1, 3:2}).astype(int)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    return report

def cv_predict_and_report(model, X, y, cv: int, seed: int = 69):
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    report = classification_report(y, y_pred, output_dict=True, digits=4)
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional path to CTG.csv")
    ap.add_argument("--model", type=str, default="lgbm",
                    help="Model: logreg | rf | lgbm | xgb | spe | smoteboost | knn")
    ap.add_argument("--cv", type=int, default=0, help="If >0, run Stratified K-Fold CV with K folds")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--save_model", type=str, default=None, help="Optional path to save fitted model (joblib)")
    ap.add_argument("--metrics_out", type=str, default=None, help="Optional path to save metrics JSON")
    args = ap.parse_args()

    X, y_raw = load_data(args.csv)
    y = encode_targets(y_raw)

    if args.cv and args.cv > 1:
        model = get_model(args.model, seed=args.seed)
        report = cv_predict_and_report(model, X, y, cv=args.cv, seed=args.seed)
        print(json.dumps(report, indent=2))
        if args.metrics_out:
            Path(args.metrics_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    X_train, X_test, y_train, y_test = make_splits(X, y, test_size=args.test_size, seed=args.seed)
    model = get_model(args.model, seed=args.seed)
    report = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Pretty print
    print("=== Classification Report (test split) ===")
    print(json.dumps(report, indent=2))

    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[✓] Wrote metrics to {args.metrics_out}")

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.save_model)
        print(f"[✓] Saved model to {args.save_model}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
