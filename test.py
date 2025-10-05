from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from ucimlrepo import fetch_ucirepo
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
lgbm = joblib.load("./models/lgbm_model.pkl")

start = time.time()
y_pred = lgbm.predict(X_test)
elasped = time.time() - start

print(f"Time taken to test: {elasped:.2f}s")
print("\nClassification Report:\n", classification_report(y_test, y_pred))