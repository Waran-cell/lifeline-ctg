import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for saving figures
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

RND = 42
ROOT = Path(".")
ART = ROOT / "ctg_rf_artifacts"
ART.mkdir(exist_ok=True, parents=True)

def load_data(path="CTG.csv"):
    df = pd.read_csv(path)
    if "NSP" not in df.columns:
        raise ValueError("CTG.csv must contain column 'NSP' with values {1,2,3}.")
    df = df.dropna(subset=["NSP"]).copy()
    y = df["NSP"].astype(int).map({1:0, 2:1, 3:2})
    drop_cols = [c for c in ["FileName","Date","SegFile"] if c in df.columns]
    X = df.drop(columns=drop_cols + ["NSP","CLASS","SUSP"], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    return X, y

def evaluate_and_plot(y_true, y_pred, labels=[0,1,2]):
    ba = float(balanced_accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(labels, labels); plt.yticks(labels, labels)
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(ART / "cm_rf.png")
    plt.close()
    metrics = {"balanced_accuracy": ba, "macro_f1": mf1}
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    rep = classification_report(y_true, y_pred, digits=3)
    with open(ART / "classification_report.txt", "w") as f:
        f.write(rep)
    return metrics

def main():
    X, y = load_data("CTG.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RND, stratify=y
    )
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=RND, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = evaluate_and_plot(y_test, y_pred)
    # Feature importance
    final_rf = None
    for name, step in rf.steps:
        if hasattr(step, "feature_importances_"):
            final_rf = step
            break
    if final_rf is not None:
        importances = final_rf.feature_importances_
        idx = np.argsort(importances)[::-1]
        plt.figure()
        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), [X.columns[i] for i in idx], rotation=90)
        plt.title("RF Feature Importance")
        plt.tight_layout()
        plt.savefig(ART / "rf_feature_importance.png")
        plt.close()
    print("Done. Artifacts saved to:", ART.resolve())
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
