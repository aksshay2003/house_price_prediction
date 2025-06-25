from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

# Load model from MLflow registry
model_uri = "models:/HousePriceModel/Production"
model = mlflow.pyfunc.load_model(model_uri)

app = Flask(__name__)

@app.route("/")
def home():
    return "House Price Prediction Model is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        df = pd.DataFrame([data])  # assuming single record
        prediction = model.predict(df)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
