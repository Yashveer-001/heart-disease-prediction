import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
model = joblib.load(open("heart_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = np.array([int_features])
        prediction = model.predict(final_features)

        output = "⚠️ High Risk of Heart Disease" if prediction[0] == 1 else "✅ Low Risk of Heart Disease"
        return render_template("index.html", prediction_text=output)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
