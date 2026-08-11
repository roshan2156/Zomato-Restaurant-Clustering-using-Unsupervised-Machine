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
        margin-top: 30px;
        margin-bottom: 25px;
    }

    .card-soft {
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

    .info-box {
        background-color: #fafafa;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #eeeeee;
        min-height: 160px;
    }

    .info-box h3 {
        color: #ef4f5f;
        font-size: 20px;
    }

    .info-box p {
        color: #666666;
        font-size: 15px;
    }

    .footer {
        text-align: center;
        color: #777777;
        padding: 30px;
        margin-top: 40px;
        font-size: 14px;
    }

    .footer-title {
        color: #ef4f5f;
        font-size: 18px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("sentiment_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


try:

    sentiment_model = load_model()

except Exception as e:

    st.error(f"Error loading sentiment model: {e}")
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
# SENTIMENT ANALYSIS
# =========================================================

st.markdown(
    """
    <div class="section-title">
        💬 Review Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)

# Customer Review

review = st.text_area(
    "Customer Review",
    placeholder="Write your review...",
    height=170
)

# =========================================================
# PREDICT
# =========================================================

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    if review.strip() == "":

        st.warning("Please enter a customer review.")

    else:

        try:

            prediction = sentiment_model.predict(
                [review]
            )[0]

            prediction = str(prediction).strip()

            # Positive
            if prediction.lower() == "positive":

                st.markdown(
                    """
                    <div class="positive">
                        😊 Positive Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Negative
            elif prediction.lower() == "negative":

                st.markdown(
                    """
                    <div class="negative">
                        😞 Negative Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Neutral
            elif prediction.lower() == "neutral":

                st.markdown(
                    """
                    <div class="neutral">
                        😐 Neutral Review
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.info(
                    f"Predicted Sentiment: {prediction}"
                )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )

# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    """
    <div class="card-soft">

        <div class="section-title">
            ⚙️ How It Works
        </div>

        <div style="
            display:flex;
            gap:20px;
            flex-wrap:wrap;
        ">

            <div class="info-box" style="flex:1;">

                <h3>
                    💬 1. Customer Review
                </h3>

                <p>
                    The user enters any restaurant
                    review into the application.
                </p>

            </div>

            <div class="info-box" style="flex:1;">

                <h3>
                    📝 2. TF-IDF
                </h3>

                <p>
                    The review text is converted
                    into numerical features using TF-IDF.
                </p>

            </div>

            <div class="info-box" style="flex:1;">

                <h3>
                    🧠 3. Prediction
                </h3>

                <p>
                    Logistic Regression predicts
                    Positive, Neutral, or Negative.
                </p>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.markdown(
    """
    <div class="card-soft">

        <div class="section-title">
            💡 Business Insights
        </div>

        <div style="
            display:flex;
            gap:20px;
            flex-wrap:wrap;
        ">

            <div class="info-box" style="flex:1;">

                <h3>
                    ⭐ Customer Experience
                </h3>

                <p>
                    Sentiment analysis helps restaurants
                    understand overall customer satisfaction
                    from reviews.
                </p>

            </div>

            <div class="info-box" style="flex:1;">

                <h3>
                    📈 Service Improvement
                </h3>

                <p>
                    Negative reviews can help identify
                    areas such as food quality, service,
                    and staff behaviour that need improvement.
                </p>

            </div>

            <div class="info-box" style="flex:1;">

                <h3>
                    💬 Review Monitoring
                </h3>

                <p>
                    Large numbers of customer reviews
                    can be automatically classified as
                    Positive, Neutral, or Negative.
                </p>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-title">
            🍽️ Zomato Restaurant Sentiment Analysis
        </div>

        <br>

        TF-IDF + Logistic Regression

    </div>
    """,
    unsafe_allow_html=True
)
