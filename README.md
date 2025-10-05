# Lifeline – Predicting Fetal Health from CTG Data
TM-214
Tan Xeng Ian
Ma Angela Sophia Isidro Rosel
K Eswaran

## 📌 Overview
**Lifeline** is a machine learning project designed to classify **fetal health status** using **Cardiotocography (CTG) data**.  
The system predicts whether a fetus is in a **Normal, Suspect, or Pathological** state, providing decision support to clinicians for early risk detection.

---

## ⚡ Features
- Uses the **UCI Cardiotocography dataset** (2,126 samples, 21 features).
- Handles **class imbalance** with balancing techniques and weighted models.
- Implements multiple ML methods:
  - Logistic Regression (L1 regularization)
  - Decision Trees & Random Forests
  - LightGBM & XGBoost
  - Self-Paced Ensemble & SMOTEBoost
  - k-Nearest Neighbors
- Includes **5-fold cross-validation** for robust evaluation.
- Delivered as Python scripts (`text.py`, `train.py`) for reproducibility.

---

## 🗂 Project Structure
```
├── text.py                  # Dataset summary and markdown report generator
├── train.py                 # Training & evaluation pipeline for multiple models
├── README.md                # Project documentation
├── requirements.txt          # Python dependencies
└── data/
    └── CTG.csv              # (Optional) Local dataset file
```

---

## 🚀 Setup & Usage

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/lifeline.git
cd lifeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Dataset Summary
```bash
# From UCI repo
python text.py --out summary.md

# Or from a local CSV
python text.py --csv ./data/CTG.csv --out summary.md
```

### 4. Train & Evaluate Models
```bash
# LightGBM (default 80/20 split)
python train.py --model lgbm

# XGBoost with 5-fold CV and save outputs
python train.py --model xgb --cv 5 --metrics_out metrics_xgb_cv.json --save_model models/xgb_model.pkl

# RandomForest baseline
python train.py --model rf

# Logistic Regression (L1)
python train.py --model logreg

# Using a local CSV
python train.py --csv ./data/CTG.csv --model lgbm
```

---



## 🚀 Setup & Usage (VS Code)

### 1. Open the Project in VS Code
Make sure the folder structure looks like:
```
lifeline/
├── text.py
├── train.py
├── requirements.txt
├── README.md
└── data/
    └── CTG.csv   (optional)
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
```

Activate it:
- **Windows (PowerShell)**:
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies (FIRST Command to Run)
```bash
pip install -r requirements.txt
```

### 4. Run Dataset Summary
```bash
# If you have CTG.csv locally
python text.py --csv ./data/CTG.csv --out summary.md

# Or fetch directly from UCI repo
python text.py --out summary.md
```

### 5. Train & Evaluate Models
```bash
# LightGBM (default 80/20 split)
python train.py --model lgbm

# XGBoost with 5-fold CV and save outputs
python train.py --model xgb --cv 5 --metrics_out metrics_xgb_cv.json --save_model models/xgb_model.pkl

# RandomForest baseline
python train.py --model rf

# Logistic Regression (L1)
python train.py --model logreg
```


## 📈 Results (Summary)
- **LightGBM** → Strong macro-F1, robust with class weights.
- **XGBoost** → Best overall generalization (5-fold CV).
- **Random Forest** → Stable baseline.
- **Challenge** → Suspect class is hardest due to overlap with Normal/Pathological.

---



---

## ⚠️ Note on LightGBM

By default, `train.py --model lgbm` uses **LightGBM**.  
If LightGBM is not installed, the script will **automatically fall back to RandomForestClassifier** and print a warning.

- If you want to install LightGBM properly, see [LIGHTGBM_INSTALL.md](LIGHTGBM_INSTALL.md) for OS-specific instructions.
- If you prefer to skip LightGBM entirely, use:
  ```bash
  pip install -r requirements_no_lgbm.txt
  python train.py --model lgbm   # will fallback to RandomForest
  ```

Other models (`xgb`, `rf`, `logreg`, etc.) will continue to work normally.


## 🔮 Future Work
- Collect additional balanced CTG data.
- Add explainability with **SHAP/LIME**.
- Build a **real-time clinical dashboard**.
- Explore **semi-supervised / transfer learning**.

---

## 👨‍💻 Authors
- **Team Lifeline** – Hackathon Project
