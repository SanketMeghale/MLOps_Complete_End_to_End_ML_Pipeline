from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

SPAM_THRESHOLD = 0.3


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/app")
def app_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"prediction": "Invalid", "confidence": 0})

    proba = model.predict_proba([message])[0]
    spam_index = list(model.classes_).index("spam")
    spam_score = proba[spam_index]

    prediction = "Spam" if spam_score >= SPAM_THRESHOLD else "Not Spam"

    return jsonify({
        "prediction": prediction,
        "confidence": round(spam_score * 100, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
