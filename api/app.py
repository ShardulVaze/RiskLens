import os
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

model = joblib.load(os.path.join(BASE_DIR, "ml_model", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "ml_model", "scaler.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "ml_model", "threshold.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "ml_model", "feature_names.pkl"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        data = {k: float(v) for k, v in data.items()}


        if "AGE" in data:
            data["DAYS_BIRTH"] = -data["AGE"] * 365
            del data["AGE"]

        if "EMPLOYMENT_YEARS" in data:
            data["DAYS_EMPLOYED"] = -data["EMPLOYMENT_YEARS"] * 365
            del data["EMPLOYMENT_YEARS"]

        if "CNT_PREV_APPLICATIONS" in data and "CNT_PREV_REJECTED" in data:
            total = data["CNT_PREV_APPLICATIONS"]
            rejected = data["CNT_PREV_REJECTED"]
            data["PREV_REJECT_RATE"] = rejected / total if total > 0 else 0

        df = pd.DataFrame(columns=feature_names)

        for k, v in data.items():
            if k in df.columns:
                df.loc[0, k] = v

        df = df.fillna(0)

        df_scaled = scaler.transform(df)

        probability = model.predict_proba(df_scaled)[0][1]

        if probability < 0.15:
            risk_band = "LOW"
            action = "Approve"
            color = "green"
        elif probability < 0.30:
            risk_band = "MEDIUM"
            action = "Manual Review"
            color = "orange"
        else:
            risk_band = "HIGH"
            action = "Reject / High Risk"
            color = "red"

        return render_template(
        "index.html",
        probability=round(float(probability) * 100, 2),
        risk_band=risk_band,
        action=action,
        color=color
          )

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
