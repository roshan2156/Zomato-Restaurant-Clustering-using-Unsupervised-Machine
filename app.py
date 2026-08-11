from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load sentiment model
with open("sentiment_model.pkl", "rb") as file:
    sentiment_model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment_result = None
    error_message = None

    if request.method == "POST":

        review = request.form.get("review", "").strip()

        if not review:
            error_message = "Please enter a customer review."

        else:
            try:
                prediction = sentiment_model.predict([review])[0]
                sentiment_result = str(prediction).strip()

            except Exception as e:
                error_message = f"Prediction Error: {e}"

    return render_template(
        "index.html",
        sentiment_result=sentiment_result,
        error_message=error_message
    )


if __name__ == "__main__":
    app.run(debug=False)
