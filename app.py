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

.main {
    background-color: #eef3f8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}

h1 {
    color: #1d3557;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

h3 {
    color: #264653;
}

p {
    font-size: 16px;
}

.stButton > button {
    background-color: #457b9d;
    color: white;
    border-radius: 8px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d3557;
    color: white;
}

div[data-baseweb="select"] {
    border-radius: 8px;
}

section[data-testid="stSidebar"] {
    background-color: #dce8f2;
}

/* Mobile Responsive */

@media (max-width: 768px) {

    h1 {
        font-size: 28px;
    }

    .block-container {
        padding-left: 6%;
        padding-right: 6%;
    }

    p {
        font-size: 14px;
    }

    .stButton > button {
        height: 45px;
        font-size: 16px;
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

# Responsive columns

col1, col2 = st.columns([1, 1])

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
