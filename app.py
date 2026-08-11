import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Zomato AI",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background-color: #f8f8f8;
}

.title {
    text-align: center;
    color: #ef4f5f;
    font-size: 45px;
    font-weight: bold;
    margin-top: 30px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 35px;
}

.card {
    background: white;
    padding: 35px;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}

.section-title {
    text-align: center;
    font-size: 27px;
    font-weight: 600;
    margin-bottom: 20px;
}

.positive {
    background: #eaf7ee;
    color: #258347;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

.negative {
    background: #fff0f0;
    color: #d93b4b;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

.neutral {
    background: #eef5fb;
    color: #3574a8;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 25px;
}

.info-box {
    background: #fafafa;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #eeeeee;
    min-height: 160px;
}

.info-box h3 {
    color: #ef4f5f;
}

.info-box p {
    color: #666;
}

.footer {
    text-align: center;
    color: #777;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():

    with open("sentiment_model.pkl", "rb") as file:
        return pickle.load(file)


try:
    sentiment_model = load_model()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Header
st.markdown(
    '<div class="title">🍽️ Zomato AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand your customer reviews using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# Sentiment Analysis
st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '💬 Review Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter any customer review to predict whether "
    "the sentiment is Positive, Neutral, or Negative."
)

review = st.text_area(
    "Customer Review",
    placeholder=(
        "Example: The food was amazing and "
        "the staff was very friendly."
    ),
    height=170
)

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a customer review.")

    else:

        try:

            prediction = sentiment_model.predict([review])[0]
            prediction = str(prediction).strip().lower()

            if prediction == "positive":

                st.markdown(
                    '<div class="positive">'
                    '😊 Positive Review'
                    '</div>',
                    unsafe_allow_html=True
                )

            elif prediction == "negative":

                st.markdown(
                    '<div class="negative">'
                    '😞 Negative Review'
                    '</div>',
                    unsafe_allow_html=True
                )

            elif prediction == "neutral":

                st.markdown(
                    '<div class="neutral">'
                    '😐 Neutral Review'
                    '</div>',
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

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# Footer
st.markdown(
    """
    <div class="footer">
        🍽️ Zomato Restaurant Sentiment Analysis
        <br><br>
        TF-IDF + Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)
