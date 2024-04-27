import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from audio_to_img import audio_to_spectrogram

# Load model
model = load_model("my_model.keras")

# my classes
class_names = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    spectrogram = audio_to_spectrogram(file.stream)

    # prediction
    prediction = model.predict(np.array([spectrogram]))

    predicted_class_index = np.argmax(prediction, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]

    confidence = np.max(prediction)

    return jsonify(
        {"prediction": predicted_class_name, "confidence": float(confidence)}
    )


if __name__ == "__main__":
    app.run(debug=False)
