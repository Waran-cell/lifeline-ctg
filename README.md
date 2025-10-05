# Lifeline CTG – Single Model (Random Forest)

Single-model solution for classifying CTG: **Random Forest**.

## Setup
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Place `CTG.csv` in the project root (contains column `NSP` with values {1,2,3}).

## Run
```bash
python ctg_single_model_rf.py
```
Artifacts will be saved to `ctg_rf_artifacts/`:
- `metrics.json` (Balanced Accuracy, Macro-F1)
- `cm_rf.png` (confusion matrix)
- `rf_feature_importance.png`
- `classification_report.txt`

