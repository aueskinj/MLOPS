from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model_path = 'california_knn_pipeline.pkl'

# Load model at startup
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON like: {"MedInc": 3.5, "HouseAge": 25.0, ... }
    input_json = request.get_json()
    if not input_json:
        return jsonify({"error": "Invalid input, expected JSON"}), 400

    # Convert to DataFrame with one row (ensure columns order matches training)
    input_df = pd.DataFrame([input_json], columns=model.named_steps['preprocessor'].transformers_[0][2])
    preds = model.predict(input_df)
    return jsonify({"prediction": float(preds[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
