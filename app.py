import streamlit as st
import google.generativeai as genai
import easyocr
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import pyttsx3

# Configure Gemini API key
genai.configure(api_key="AIzaSyCE55K6hzrSvxuCm6de_feQkUSagAqueOM")  # Replace with your Gemini API Key

# EasyOCR Reader
reader = easyocr.Reader(['en'])

# Gemini AI configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Global variable for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction Bot", layout="wide")

# Add background image
image_file = "pexels-karolina-grabowska-4226764.jpg"  # Ensure this file is in the same folder as your script
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("file://{os.path.abspath(image_file)}");
        background-size: cover;
        background-blend-mode: overlay;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("❤️ Heart Disease Prediction Bot")
st.markdown("Analyze your heart health by providing input data and uploading relevant health or ECG data images.")

# Sidebar Inputs
st.sidebar.header("Input Your Health Data")
age = st.sidebar.slider("Age (years):", 0, 100, 50)
gender = st.sidebar.selectbox("Gender:", ["Male", "Female", "Other"])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type:", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.sidebar.slider("Resting Blood Pressure (mmHg):", 50, 200, 120)
cholesterol = st.sidebar.slider("Cholesterol Level (mg/dL):", 100, 400, 200)
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL:", ["Yes", "No"])
resting_ecg = st.sidebar.selectbox("Resting ECG Results:", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_heart_rate = st.sidebar.slider("Max Heart Rate Achieved (bpm):", 60, 220, 150)
exercise_induced_angina = st.sidebar.selectbox("Exercise-Induced Angina:", ["Yes", "No"])
st_depression = st.sidebar.slider("ST Depression Induced by Exercise:", 0.0, 5.0, 1.0)

# Submit Button
if st.sidebar.button("Analyze"):
    # Collect input details
    details = {
        "Age": age,
        "Gender": gender,
        "Chest Pain Type": chest_pain_type,
        "Resting Blood Pressure": resting_bp,
        "Cholesterol": cholesterol,
        "Fasting Blood Sugar > 120 mg/dL": fasting_blood_sugar,
        "Resting ECG Results": resting_ecg,
        "Max Heart Rate Achieved": max_heart_rate,
        "Exercise-Induced Angina": exercise_induced_angina,
        "ST Depression Induced by Exercise": st_depression,
    }

    # Use AI for heart disease prediction
    prompt = (
        f"The user has provided the following health details for heart disease prediction:\n"
        f"{details}\n"
        f"Provide a detailed analysis of the likelihood of heart disease and suggestions for improving heart health."
    )

    try:
        if "chat_session" not in st.session_state or st.session_state.chat_session is None:
            st.session_state.chat_session = model.start_chat(history=[{
                "role": "user", 
                "parts": ["You are a health AI specialized in heart disease prediction and prevention."]
            }])

        response = st.session_state.chat_session.send_message(prompt)
        st.session_state.chat_history.append({"inputs": prompt, "bot_response": response.text})

        # Display AI Insights
        st.subheader("Heart Disease Prediction Analysis")
        st.markdown("### AI Insights:")
        st.write(response.text)

        # Voice Output
        engine = pyttsx3.init()
        engine.say(response.text)
        engine.runAndWait()

        # Risk Distribution Visualization
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "Medium Risk", "High Risk"], [30, 50, 20], color=['green', 'orange', 'red'])
        ax.set_title("Heart Disease Risk Levels")
        ax.set_ylabel("Probability (%)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Chat Section
st.subheader("Chat with the Bot")
with st.form("chat_form"):
    user_message = st.text_input("Your Question:", placeholder="Type your question here...")
    uploaded_chat_file = st.file_uploader("Upload ECG or Heart Health Data Image (optional):", type=["jpg", "jpeg", "png", "pdf"])
    submitted = st.form_submit_button("Send")

    if submitted:
        if user_message:
            try:
                if "chat_session" not in st.session_state or st.session_state.chat_session is None:
                    st.session_state.chat_session = model.start_chat(history=[{
                        "role": "user", 
                        "parts": ["You are a health AI specialized in heart disease prediction and prevention."]
                    }])
                response = st.session_state.chat_session.send_message(user_message)
                st.session_state.chat_history.append({"inputs": user_message, "bot_response": response.text})
                st.markdown(f"**You:** {user_message}")
                st.markdown(f"**Bot:** {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif uploaded_chat_file:
            os.makedirs("uploads", exist_ok=True)
            file_path = os.path.join("uploads", secure_filename(uploaded_chat_file.name))
            with open(file_path, "wb") as f:
                f.write(uploaded_chat_file.read())

            if uploaded_chat_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                extracted_text = reader.readtext(file_path, detail=0)
                if extracted_text:
                    extracted_text_combined = " ".join(extracted_text)
                    st.success("Text extracted from the image:")
                    st.text(extracted_text_combined)

                    prompt = (
                        f"The following text was extracted from the uploaded file:\n"
                        f"\"{extracted_text_combined}\"\n"
                        f"Provide insights on heart health and potential risks based on this data."
                    )
                    try:
                        if "chat_session" not in st.session_state or st.session_state.chat_session is None:
                            st.session_state.chat_session = model.start_chat(history=[{
                                "role": "user", 
                                "parts": ["You are a health AI specialized in heart disease prediction and prevention."]
                            }])

                        response = st.session_state.chat_session.send_message(prompt)
                        st.session_state.chat_history.append({"inputs": prompt, "bot_response": response.text})
                        st.markdown(f"**Bot Suggestions:** {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No text found in the uploaded image.")
            else:
                st.warning("Unsupported file type for analysis.")

# Display Chat History
if st.session_state.chat_history:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**Input:** {chat['inputs']}")
        st.markdown(f"**Bot Response:** {chat['bot_response']}")
