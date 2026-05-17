import os
import pandas as pd
import joblib
from flask import Flask, request, render_template

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
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

        for k, v in data.items():
            try:
                float(v)
            except ValueError:
                return render_template("index.html", error=f"Invalid value for {k}: must be a number.")

        data = {k: float(v) for k, v in data.items()}

        age = data.get("AGE", 0)
        emp = data.get("EMPLOYMENT_YEARS", 0)

        if age <= 0 or age > 100:
            return render_template("index.html", error="Age must be between 1 and 100.")
        if emp < 0 or emp > age:
            return render_template("index.html", error="Employment years can't be negative or exceed age.")

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

        df = pd.DataFrame([data], columns=feature_names)
        df = df.fillna(0)

        df_scaled = scaler.transform(df)

        probability = model.predict_proba(df_scaled)[0][1]

        low_cutoff = threshold * 0.5
        if probability < 0.15:
         risk_band, action, color = "LOW", "Approve", "green"
        elif probability < 0.35:
         risk_band, action, color = "MEDIUM", "Manual Review", "orange"
        else:
          risk_band, action, color = "HIGH", "Reject / High Risk", "red"

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