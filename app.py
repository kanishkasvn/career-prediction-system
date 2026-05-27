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

/* Rich background gradient */
.main {
    background: linear-gradient(135deg, #f6f8fb 0%, #e9effd 100%);
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 5%;
    padding-right: 5%;
}

/* Chromatic multi-stop gradient for main title */
h1 {
    background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

h3 {
    color: #1e1b4b;
    font-weight: 700;
    font-size: 24px;
    border-bottom: 3px solid #6366f1;
    padding-bottom: 8px;
    margin-bottom: 25px;
}

p {
    font-size: 16px;
    color: #475569;
}

/* Vivid background with glow for the action button */
.stButton > button {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    color: white !important;
    border-radius: 10px;
    height: 52px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
    transform: translateY(-1px);
}

/* Colorful input field matching */
div[data-baseweb="select"] {
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}

/* --- OVERRIDE DEFAULT RED SLIDERS TO MATCH THEME --- */
div[data-testid="stSlider"] [data-testid="stThumb"] {
    background-color: #8b5cf6 !important;
    border: 2px solid #3b82f6 !important;
}

div[data-testid="stSlider"] [data-base-class="stSlider"] div {
    background-color: #3b82f6 !important;
}

div[data-testid="stSlider"] > div {
    padding-bottom: 15px;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    h1 {
        font-size: 30px;
    }
    .block-container {
        padding-left: 6%;
        padding-right: 6%;
    }
    p {
        font-size: 14px;
    }
    .stButton > button {
        height: 48px;
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
