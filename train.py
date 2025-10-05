from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import joblib
import time

cardiotocography = fetch_ucirepo(id=193)
X = cardiotocography.data.features
y = cardiotocography.data.targets
y = y["NSP"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=47
)

y_train -= 1
y_test -= 1

lgbm = LGBMClassifier(
    n_estimators=110,
    learning_rate=0.05,
    max_depth=-1,
    random_state=43,
    class_weight="balanced",
    verbose=-1   
)

start = time.time()
lgbm.fit(X_train, y_train)
elasped = time.time() - start 

print(f"Time taken to train: {elasped:.2f}s")
print("Model trained successfully. Saving model...")
joblib.dump(lgbm, "./models/lgbm_model.pkl")
