import streamlit as st
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Zomato Sentiment Analysis",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8f8f8;
    }

    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 700;
        color: #ef4f5f;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 40px;
    }

    .section-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        color: #222222;
        margin-bottom: 25px;
    }

    .card {
        background-color: white;
        padding: 35px;
        border-radius: 30px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.06);
        margin-bottom: 30px;
    }

    .positive {
        background-color: #eaf7ee;
        color: #258347;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
    }

    .negative {
        background-color: #fff0f0;
        color: #d93b4b;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
    }

    .neutral {
        background-color: #eef5fb;
        color: #3574a8;
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
    }

    div.stButton > button {
        background-color: #ef4f5f;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-size: 17px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #d93b4b;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD SENTIMENT MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("sentiment_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


# =========================================================
# LOAD MODEL
# =========================================================

try:

    sentiment_model = load_model()

except Exception as e:

    st.error(
        f"Error loading sentiment model: {e}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🍽️ Zomato AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Understand your customer reviews using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SENTIMENT ANALYSIS CARD
# =========================================================

st.markdown(
    """
    <div class="card">
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-title">
        💬 Review Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter any customer review to predict whether "
    "the sentiment is Positive, Neutral, or Negative."
)


# =========================================================
# CUSTOMER REVIEW
# =========================================================

review = st.text_area(
    "Customer Review",
    placeholder=(
        "Write your review... "
        "Example: The food was amazing and "
        "the staff was very friendly."
    ),
    height=170
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    # -----------------------------------------------------
    # CHECK EMPTY REVIEW
    # -----------------------------------------------------

    if review.strip() == "":

        st.warning(
            "Please enter a customer review."
        )

    else:

        try:

            # -------------------------------------------------
            # MODEL PREDICTION
            # -------------------------------------------------

            prediction = sentiment_model.predict(
                [review]
            )[0]

            prediction = str(
                prediction
            ).strip().lower()


            # -------------------------------------------------
            # POSITIVE
            # -------------------------------------------------

            if prediction == "positive":

                st.markdown(
                    """
                    <div class="positive">
                        😊 Positive Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # NEGATIVE
            # -------------------------------------------------

            elif prediction == "negative":

                st.markdown(
                    """
                    <div class="negative">
                        😞 Negative Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # NEUTRAL
            # -------------------------------------------------

            elif prediction == "neutral":

                st.markdown(
                    """
                    <div class="neutral">
                        😐 Neutral Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # OTHER OUTPUT
            # -------------------------------------------------

            else:

                st.info(
                    f"Predicted Sentiment: {prediction}"
                )


        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )


# =========================================================
# CLOSE CARD
# =========================================================

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)