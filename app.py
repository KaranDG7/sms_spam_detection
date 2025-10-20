from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# === 1. Load model and tokenizer ===
model = tf.keras.models.load_model("sms_fraud_tf_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# === 2. Constants ===
vocab_size = 8000
max_len = 50

# === 3. Flask app ===
app = Flask(__name__)

# Prediction function
def predict_sms(message):
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    label = "SPAM" if pred > 0.5 else "HAM"
    return label, float(pred)

# Home route
@app.route("/", methods=["GET"])
def home():
    return "SMS Spam Detection API is running!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    sms_text = request.form.get("sms")
    if not sms_text:
        return jsonify({"error": "Please provide 'sms' in form data"}), 400

    label, probability = predict_sms(sms_text)
    return jsonify({'result': label, 'probability': probability})


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
