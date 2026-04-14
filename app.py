from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# =========================
# 🤖 LOAD TRAINED MODEL
# =========================

print("🤖 Loading trained model...")

classifier = pipeline(
    "text-classification",
    model="./fake_news_model",
    tokenizer="./fake_news_model"
)

print("✅ Model loaded successfully!")

# =========================
# 🏠 HOME ROUTE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# 🔍 PREDICT ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # 🤖 Model prediction
        result = classifier(text)[0]

        # Convert label
        label = "Fake" if result['label'] == 'LABEL_0' else "Real"

        # Confidence (0–1)
        confidence = float(result['score'])

        return jsonify({
            "label": label,
            "score": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# ▶️ RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)