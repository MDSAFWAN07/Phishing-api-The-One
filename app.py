from flask import Flask, request, jsonify
import re, json, joblib, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# === Load Models ===
model_path = "cnn_lstm_feature_extractor_v4.h5"
xgb_path = "xgboost_classifier_v4.pkl"
scaler_path = "scaler.pkl"
features_path = "feature_names.json"

if not (os.path.exists(model_path) and os.path.exists(xgb_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
    raise FileNotFoundError("❌ Missing model/scaler files. Please add your 4 trained model files to this folder before running.")

cnn_lstm = load_model(model_path, compile=False)
deep_model = tf.keras.Model(inputs=cnn_lstm.input, outputs=cnn_lstm.get_layer("deep_features").output)
xgb = joblib.load(xgb_path)
scaler = joblib.load(scaler_path)
feature_names = json.load(open(features_path))

def extract_features(url):
    url = str(url).lower()
    suspicious = ['login','verify','update','secure','account','bank','payment','signin']
    domain = re.findall(r'https?://([^/]+)', url)
    domain = domain[0] if domain else url
    return np.array([
        len(url), url.count('.'), url.count('-'),
        1 if '@' in url else 0, url.count('/'),
        1 if 'www' in url else 0, 1 if '.com' in url else 0,
        1 if url.startswith('https') else 0,
        len(url.split('.')[-1]) if '.' in url else 0,
        1 if re.match(r"http[s]?://\\d", url) else 0,
        url.count('.') - 1,
        1 if any(k in url for k in suspicious) else 0,
        1 if 'https' in domain else 0,
        len(domain),
        sum(c.isdigit() for c in url)
    ])

@app.route('/')
def home():
    return jsonify({"message": "✅ Phishing Detection API is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "URL not provided"}), 400

        feats = extract_features(url).reshape(1, -1)
        scaled = scaler.transform(feats)
        deep_feats = deep_model.predict(scaled[..., None], verbose=0)
        hybrid_feats = np.concatenate([deep_feats, scaled], axis=1)
        prob = xgb.predict_proba(hybrid_feats)[0][1]
        label = "Phishing" if prob >= 0.45 else "Legitimate"

        return jsonify({
            "url": url,
            "prediction": label,
            "confidence": round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)