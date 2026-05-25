import streamlit as st
import numpy as np
import pickle
import joblib

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="Career Prediction System",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

/* Main Background */

.main {
    background: linear-gradient(
        135deg,
        #0b1f3a,
        #102b50,
        #163d6b
    );
    color: white;
}

/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}

/* Title */

h1 {
    color: #ffffff;
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    font-family: 'Trebuchet MS', sans-serif;
    letter-spacing: 1px;
}

/* Subheadings */

h3 {
    color: #d6e4ff;
    font-family: 'Verdana', sans-serif;
}

/* Paragraph text */

p {
    color: #e8eef7;
    font-size: 16px;
    font-family: 'Segoe UI', sans-serif;
}

/* Labels */

label {
    color: #ffffff !important;
    font-weight: 500;
    font-family: 'Segoe UI', sans-serif;
}

/* Selectbox styling */

div[data-baseweb="select"] {
    background-color: #f5f7fa;
    border-radius: 8px;
    color: black;
}

/* Slider text */

.stSlider {
    color: white;
}

/* Radio button text */

.stRadio label {
    color: white !important;
}

/* Button */

.stButton > button {
    background-color: #4ea8de;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #90e0ef;
    color: #001233;
}

/* Success box */

.stSuccess {
    background-color: #caf0f8;
    color: #001233;
    border-radius: 10px;
    padding: 10px;
}

/* Info box */

.stInfo {
    background-color: #d6e4ff;
    color: #001233;
    border-radius: 10px;
}

/* Horizontal line */

hr {
    border: 1px solid #4ea8de;
}

/* Mobile Responsive */

@media (max-width: 768px) {

    h1 {
        font-size: 28px;
    }

    p {
        font-size: 14px;
    }

    .block-container {
        padding-left: 6%;
        padding-right: 6%;
    }

    .stButton > button {
        font-size: 16px;
        height: 45px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("compressed_model.pkl")

encoders = pickle.load(open("encoders.pkl", "rb"))

# ---------------- TITLE ---------------- #

st.title("Career Prediction System")

st.write(
    "This system predicts suitable career paths based on "
    "technical skills, certifications, interests and personality traits."
)

st.markdown("---")

# ---------------- INPUT SECTION ---------------- #

st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

# ---------- LEFT COLUMN ---------- #

with col1:

    logical_rating = st.slider(
        "Logical Quotient Rating",
        1,
        10,
        5
    )

    coding_skills = st.slider(
        "Coding Skills Rating",
        1,
        10,
        5
    )

    hackathons = st.slider(
        "Hackathons Participated",
        0,
        10,
        0
    )

    public_speaking = st.slider(
        "Public Speaking Confidence",
        1,
        10,
        5
    )

    management_skills = st.slider(
        "Management Skills",
        1,
        10,
        5
    )

    reading_skills = st.selectbox(
        "Reading & Writing Skills",
        ["poor", "medium", "excellent"]
    )

    memory_capability = st.selectbox(
        "Memory Capability",
        ["poor", "medium", "excellent"]
    )

# ---------- RIGHT COLUMN ---------- #

with col2:

    interested_subjects = st.selectbox(
        "Primary Area of Interest",
        [
            "programming",
            "Management",
            "data engineering",
            "networks",
            "Software Engineering",
            "cloud computing",
            "parallel computing",
            "IOT",
            "hacking",
            "Computer Architecture",
            "Data Science"
        ]
    )

    certifications = st.selectbox(
        "Certification Domain",
        [
            "Machine Learning",
            "App Development",
            "Python",
            "Shell Programming",
            "R Programming",
            "Information Security",
            "Hadoop",
            "Full Stack",
            "Distro Making",
            "Testing"
        ]
    )

    workshops = st.selectbox(
        "Most Impactful Workshop",
        [
            "Data Science",
            "Testing",
            "Game Development",
            "Cloud Computing",
            "Web Technologies",
            "Database Security",
            "System Designing",
            "Hacking",
            "Machine Learning",
            "Application Development"
        ]
    )

    career_interest = st.selectbox(
        "Preferred Career Stream",
        [
            "BPA",
            "Cloud Services",
            "Testing and QA",
            "Sales and Marketing",
            "System Developer",
            "Business process analyst",
            "Developer",
            "Security",
            "Cloud Engineer",
            "Data Science"
        ]
    )

    work_style = st.radio(
        "Preferred Work Style",
        ["Technical", "Management"],
        horizontal=True
    )

    introvert = st.radio(
        "Are you an Introvert?",
        ["No", "Yes"],
        horizontal=True
    )

    self_learning = st.radio(
        "Self Learning Capability",
        ["Yes", "No"],
        horizontal=True
    )

# ---------------- ENCODING ---------------- #

reading_encoded = encoders[
    "reading and writing skills"
].transform([reading_skills])[0]

memory_encoded = encoders[
    "memory capability score"
].transform([memory_capability])[0]

introvert_encoded = encoders[
    "Introvert"
].transform([introvert.lower()])[0]

self_learning_encoded = encoders[
    "self-learning capability?"
].transform([self_learning.lower()])[0]

# manual encoding

work_style_encoded = 1 if work_style == "Management" else 0

# ---------------- PREDICTION ---------------- #

st.markdown("---")

if st.button("Predict Career Path"):

    input_data = np.array([[

        logical_rating,
        hackathons,
        coding_skills,
        public_speaking,

        self_learning_encoded,

        0,

        reading_encoded,
        memory_encoded,

        work_style_encoded,
        management_skills,

        0,
        0,
        0,

        0,

        0,
        0,
        0,

        1,
        introvert_encoded

    ]])

    prediction = model.predict(input_data)

    try:

        final_prediction = encoders[
            "Suggested Job Role"
        ].inverse_transform(prediction)

        st.success(
            "Suggested Career Path : " +
            str(final_prediction[0])
        )

    except:

        st.success(
            "Predicted Career Code : " +
            str(prediction[0])
        )

    st.info("Prediction generated successfully.")
