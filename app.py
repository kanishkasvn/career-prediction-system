import streamlit as st
import numpy as np
import pickle
import joblib

# page settings

st.set_page_config(
    page_title="Career Prediction System",
    layout="centered"
)

# custom styling

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
    font-size: 42px;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #125d91;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

div[data-baseweb="select"] {
    border-radius: 8px;
}

.css-1d391kg {
    background-color: #e8eef5;
}

</style>
""", unsafe_allow_html=True)

# loading files

model = joblib.load("compressed_model.pkl")

encoders = pickle.load(open("encoders.pkl", "rb"))

# title section

st.title("Career Prediction System")

st.write(
    "This system predicts suitable career roles based on student skills, interests and personality traits."
)

st.markdown("---")

# input section

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:

    logical_rating = st.slider(
        "Logical Quotient Rating",
        0,
        10,
        5
    )

    coding_skills = st.slider(
        "Coding Skills Rating",
        0,
        10,
        5
    )

    public_speaking = st.slider(
        "Public Speaking Points",
        0,
        10,
        5
    )

    hackathons = st.slider(
        "Hackathons Participated",
        0,
        10,
        2
    )

    self_learning = st.selectbox(
        "Self Learning Capability",
        ["yes", "no"]
    )

    extra_courses = st.selectbox(
        "Extra Courses Completed",
        ["yes", "no"]
    )

with col2:

    worked_in_teams = st.selectbox(
        "Worked in Teams",
        ["yes", "no"]
    )

    introvert = st.selectbox(
        "Introvert",
        ["yes", "no"]
    )

    reading_skills = st.selectbox(
        "Reading and Writing Skills",
        ["poor", "medium", "excellent"]
    )

    memory_capability = st.selectbox(
        "Memory Capability Score",
        ["poor", "medium", "excellent"]
    )

    smart_worker = st.selectbox(
        "Smart Worker",
        ["yes", "no"]
    )

    management_skills = st.slider(
        "Management Skills",
        0,
        10,
        5
    )

# encoding values

self_learning_encoded = encoders[
    "self-learning capability?"
].transform([self_learning])[0]

extra_courses_encoded = encoders[
    "Extra-courses did"
].transform([extra_courses])[0]

worked_in_teams_encoded = encoders[
    "worked in teams ever?"
].transform([worked_in_teams])[0]

introvert_encoded = encoders[
    "Introvert"
].transform([introvert])[0]

reading_encoded = encoders[
    "reading and writing skills"
].transform([reading_skills])[0]

memory_encoded = encoders[
    "memory capability score"
].transform([memory_capability])[0]

smart_worker_encoded = 1 if smart_worker == "yes" else 0

# prediction

if st.button("Predict Career"):

    input_data = np.array([[

        logical_rating,
        hackathons,
        coding_skills,
        public_speaking,

        self_learning_encoded,
        extra_courses_encoded,

        reading_encoded,
        memory_encoded,

        smart_worker_encoded,
        management_skills,

        0,
        0,
        0,

        0,

        0,
        0,
        0,

        worked_in_teams_encoded,
        introvert_encoded

    ]])

    prediction = model.predict(input_data)

    # converting prediction into original label

    try:

        final_prediction = encoders[
            "Suggested Job Role"
        ].inverse_transform(prediction)

        st.success(
            "Suggested Career Role: " +
            str(final_prediction[0])
        )

    except:

        st.success(
            "Predicted Career Code: " +
            str(prediction[0])
        )

    st.info("Prediction generated successfully.")
