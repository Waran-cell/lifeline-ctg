# Lifeline – Predicting Fetal Health from CTG Data

TM-214
Tan Xeng Ian
Ma Angela Sophia Isidro Rosel
K Eswaran

## Overview

**Lifeline** is a machine learning project designed to classify **fetal health status** using **Cardiotocography (CTG) data**.  
The system predicts whether a fetus is in a **Normal, Suspect, or Pathological** state, providing decision support to clinicians for early risk detection.

---

## Features

- Uses the **UCI Cardiotocography dataset** (2,126 samples, 21 features).
- Handles **class imbalance** with balancing techniques and weighted models.
- Experiments with multiple ML methods, some of which:
  - Logistic Regression (L1 regularization)
  - TabNet
  - Neural network with focal loss
  - Decision Trees & Random Forests
  - LightGBM & XGBoost
  - Self-Paced Ensemble & SMOTEBoost
  - k-Nearest Neighbors
- Includes **5-fold cross-validation** for robust evaluation.
- Delivered as Python scripts (`text.py`, `train.py`) for reproducibility.

---

## Project Structure

```
├── models                            # Saved models
├── Lifeline.ipynb                    # Data exploration, experiments and model selection
├── MLDA 2025 Academic Report.pdf     # Project report
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── train.py                          # Training LightGBM model then saving the model to models.pkl
├── test.py                           # Evaluation of LightGBM model
```

---

## Setup & Usage

_Note_: This section does not account for the running of `Lifeline.ipynb`. To run `Lifeline.ipynb`, you will need to install additional packages. Optionally, you can run `Lifeline.ipynb` in Google Colab.

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/lifeline.git
cd lifeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train LGBM Model

```bash
# LightGBM (default 80/20 split)
python train.py
```

### 5. Test LGBM Model

```bash
# LightGBM (default 80/20 split)
python test.py
```
