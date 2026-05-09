import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,roc_auc_score,recall_score,f1_score,ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

print("Start")
X_train, y_train, X_test, y_test = joblib.load(
    "data/processed/train_test_data.pkl"
)

selector = SelectFromModel(XGBClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
))
selector.fit(X_train, y_train)

X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print(f"Selected top {X_train_sel.shape[1]} features")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


model = XGBClassifier(
    n_estimators=900,
    max_depth=9,
    learning_rate=0.02,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=1.5,
    min_child_weight=2,
    reg_lambda=2.5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_sel, y_train)

y_proba = model.predict_proba(X_test_sel)[:, 1]
y_pred = (y_proba > 0.20).astype(int)  

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\nModel Performance:")
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.show()

joblib.dump(model, "ml_model/best_model.pkl")
print("Done")
