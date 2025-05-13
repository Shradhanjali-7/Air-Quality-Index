from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load model and scaler at startup
model = joblib.load("models/ran_model.pkl")     # Your trained ML model
scaler = joblib.load("models/ran_scaler.pkl")   # Your fitted scaler

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result')
def result():
    return render_template("result.html")

def get_health_advisory(aqi):
    """Return AQI category and health advisory based on AQI value."""
    if aqi <= 50:
        return ("Good", "Air quality is satisfactory. It's a great day to be outside!")
    elif aqi <= 100:
        return ("Moderate", "Air quality is acceptable. Some pollutants may be a concern for sensitive individuals.")
    elif aqi <= 150:
        return ("Unhealthy for Sensitive Groups", "Children, elderly, and people with respiratory problems should limit outdoor activity.")
    elif aqi <= 200:
        return ("Unhealthy", "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion.")
    elif aqi <= 300:
        return ("Very Unhealthy", "Health alert: everyone may experience more serious health effects. Avoid outdoor activity.")
    else:
        return ("Hazardous", "Serious health effects. Stay indoors and avoid all physical activity outside.")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Extract query parameters
        features = [
            float(request.args.get("PM25", 0)),
            float(request.args.get("PM10", 0)),
            float(request.args.get("NO2", 0)),
            float(request.args.get("SO2", 0)),
            float(request.args.get("CO", 0)),
            float(request.args.get("O3", 0))
        ]

        # Scale and predict
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        aqi = round(prediction[0])

        # Get health advisory
        category, advisory = get_health_advisory(aqi)

        return jsonify({
            "aqi": aqi,
            "category": category,
            "advisory": advisory
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
