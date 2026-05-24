import streamlit as st
import numpy as np
import pickle

# loading saved files

import joblib

model = joblib.load("compressed_model.pkl")

encoders = pickle.load(open("encoders.pkl", "rb"))

# title

st.title("Career Prediction System")

st.write("Predict suitable job role based on student skills")

# taking inputs

logical_rating = st.slider(
    "Logical quotient rating",
    0,
    10,
    5
)

hackathons = st.slider(
    "Hackathons completed",
    0,
    10,
    2
)

coding_skills = st.slider(
    "Coding skills rating",
    0,
    10,
    5
)

public_speaking = st.slider(
    "Public speaking points",
    0,
    10,
    5
)

self_learning = st.selectbox(
    "Self learning capability",
    ["yes", "no"]
)

extra_courses = st.selectbox(
    "Extra courses completed",
    ["yes", "no"]
)

team_work = st.selectbox(
    "Worked in team before",
    ["yes", "no"]
)

introvert = st.selectbox(
    "Are you introvert?",
    ["yes", "no"]
)

# converting text values

self_learning = encoders[
    "self-learning capability?"
].transform([self_learning])[0]

extra_courses = encoders[
    "Extra-courses did"
].transform([extra_courses])[0]

team_work = encoders[
    "worked in teams ever?"
].transform([team_work])[0]

introvert = encoders[
    "Introvert"
].transform([introvert])[0]

# predict button

if st.button("Predict Job Role"):

    input_data = np.array([[
        logical_rating,
        hackathons,
        coding_skills,
        public_speaking,
        self_learning,
        extra_courses,

        0,
        0,
        0,
        0,
        0,
        0,
        0,

        0,

        0,
        0,
        0,

        team_work,
        introvert
    ]])

    prediction = model.predict(input_data)

    st.success(
        "Suggested Job Role : " + str(prediction[0])
    )

    st.write("Prediction completed successfully")