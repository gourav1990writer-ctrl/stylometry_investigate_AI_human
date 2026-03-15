import streamlit as st
import joblib
from SRC.features import get_basic_features
from SRC.save_feedback import save_feedback

# Load model
model = joblib.load("models/human_ai_model.joblib")

st.title("Stylometry Investigate App")
st.write("Paste a text below and the app will guess whether it looks human or AI-written.")

user_text = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        features = get_basic_features(user_text)

        # Save values in session state so buttons still know the last prediction
        st.session_state["last_text"] = user_text
        st.session_state["last_prediction"] = prediction

        st.subheader("Prediction")
        st.write(f"**Label:** {prediction}")

        st.subheader("Confidence")
        for label, prob in zip(model.classes_, probabilities):
            st.write(f"{label}: {prob:.3f}")

        st.subheader("Basic Language Features")
        st.write(f"Word count: {features['word_count']}")
        st.write(f"Sentence count: {features['sentence_count']}")
        st.write(f"Average sentence length: {features['avg_sentence_length']}")
        st.write(f"Lexical diversity: {features['lexical_diversity']}")

# Show feedback buttons only if a prediction has already been made
if "last_text" in st.session_state and "last_prediction" in st.session_state:
    st.subheader("Was this prediction correct?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Prediction correct"):
            save_feedback(
                st.session_state["last_text"],
                st.session_state["last_prediction"],
                "correct"
            )
            st.success("Thank you. Your feedback was saved.")

    with col2:
        if st.button("❌ Prediction wrong"):
            save_feedback(
                st.session_state["last_text"],
                st.session_state["last_prediction"],
                "wrong"
            )
            st.success("Thank you. Your feedback was saved.")