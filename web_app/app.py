import os
import pickle
import threading
import webbrowser

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODELS = {}
for sex in ["male", "female"]:
    model_path = os.path.join(MODELS_DIR, f"model_{sex}.pkl")
    with open(model_path, "rb") as f:
        MODELS[sex] = pickle.load(f)

def get_category(prediction: float, sex: str) -> str:
    if sex == "male":
        if prediction < 6:
            return "Essential fat"
        elif prediction < 14:
            return "Athletic"
        elif prediction < 18:
            return "Fitness"
        elif prediction < 25:
            return "Average"
        else:
            return "Above average"
    else:
        if prediction < 14:
            return "Essential fat"
        elif prediction < 21:
            return "Athletic"
        elif prediction < 25:
            return "Fitness"
        elif prediction < 32:
            return "Average"
        else:
            return "Above average"

@app.route("/")
def index():
    return render_template(
        "index.html",
        male_features=MODELS["male"]["features"],
        female_features=MODELS["female"]["features"],
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sex = data.get("sex", "male").lower()

    if sex not in MODELS:
        return jsonify({"error": "Invalid sex value"}), 400

    pipeline = MODELS[sex]["pipeline"]
    features = MODELS[sex]["features"]

    try:
        values = {feat: float(data["measurements"][feat]) for feat in features}
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Missing or invalid measurement values"}), 400

    X_new = pd.DataFrame([values])[features]
    prediction = float(pipeline.predict(X_new)[0])
    category = get_category(prediction, sex)

    return jsonify({
        "prediction": round(prediction, 1),
        "category": category,
    })

if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, port=5000, use_reloader=False)