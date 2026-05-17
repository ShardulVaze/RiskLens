import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score

print("START")
X_train, y_train, X_test, y_test = joblib.load(
    "data/processed/train_test_data.pkl"
)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

print("Training...")
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

print("\nThreshold Analysis:")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 48)

from sklearn.metrics import precision_score, recall_score

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.10, 0.60, 0.02):
    y_pred_t = (y_proba > threshold).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t)
    f = f1_score(y_test, y_pred_t)
    print(f"{threshold:<12.2f} {p:<12.4f} {r:<12.4f} {f:<12.4f}")
    if f > best_f1:
        best_f1 = f
        best_threshold = threshold

print(f"\nBest Threshold (max F1) = {round(best_threshold, 2)}")

y_pred = (y_proba > best_threshold).astype(int)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {round(roc_auc, 4)}")

joblib.dump(model, "ml_model/model.pkl")
joblib.dump(best_threshold, "ml_model/threshold.pkl")
print(f"\nmodel saved, threshold = {round(best_threshold, 2)}")