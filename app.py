# --------------------------
# app.py - Flask Application for Soil Deficiency Prediction
# --------------------------

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")

# --------------------------
# Build correct model path
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "notebook", "model")

trained_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
region_encoder_path = os.path.join(MODEL_DIR, "region_encoder.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# --------------------------
# Safe Load Model & Encoders
# --------------------------
model = joblib.load(trained_model_path)
le_deficiency = joblib.load(label_encoder_path)
le_region = joblib.load(region_encoder_path)
scaler = joblib.load(scaler_path)

# --------------------------
# Website Pages
# --------------------------
@app.route('/')
def home():
    return render_template("start.html")

@app.route('/start')
def start():
    regions = le_region.classes_
    return render_template("index.html", regions=regions)

@app.route('/minerals')
def minerals():
    return render_template("mineral.html")

@app.route('/diseases')
def diseases():
    return render_template("diseases.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/download')
def download():
    return render_template("download.html")

# --------------------------
# Soil Analysis Route
# --------------------------
@app.route('/analyze_soil', methods=['POST'])
def analyze_soil():
    try:
        pH = float(request.form['pH'])
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Calcium = float(request.form['Calcium'])
        Magnesium = float(request.form['Magnesium'])
        Iron = float(request.form['Iron'])
        Zinc = float(request.form['Zinc'])
        Copper = float(request.form['Copper'])
        Boron = float(request.form['Boron'])
        Organic_Matter = float(request.form['Organic_Matter'])
        Region = request.form['Region']

        region_encoded = le_region.transform([Region])[0]

        input_features = np.array([[pH, Nitrogen, Phosphorus, Potassium, Calcium,
                                    Magnesium, Iron, Zinc, Copper, Boron,
                                    Organic_Matter, region_encoded]])

        input_scaled = scaler.transform(input_features)

        prediction_encoded = model.predict(input_scaled)[0]
        prediction = le_deficiency.inverse_transform([prediction_encoded])[0]

        return render_template(
            'result.html',
            pH=pH, Nitrogen=Nitrogen, Phosphorus=Phosphorus, Potassium=Potassium,
            Calcium=Calcium, Magnesium=Magnesium, Iron=Iron, Zinc=Zinc, Copper=Copper,
            Boron=Boron, Organic_Matter=Organic_Matter, Region=Region,
            Predicted_Deficiency=prediction
        )

    except Exception as e:
        return f"Error occurred: {e}"

# --------------------------
# Run the app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
