import streamlit as st

from predict import predict_fraud

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")

EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract", "Temporary", "Unknown"]
EXPERIENCE_LEVELS = [
    "Not Applicable",
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
    "Unknown",
]
EDUCATION_LEVELS = [
    "Unspecified",
    "High School or equivalent",
    "Some High School Coursework",
    "Vocational - HS Diploma",
    "Some College Coursework Completed",
    "Associate Degree",
    "Bachelor's Degree",
    "Master's Degree",
    "Doctorate",
    "Certification",
    "Professional",
    "Vocational",
    "Vocational - Degree",
]

st.title("🕵️ Fake Job Posting Detector")
st.write(
    "Fill in the details of a job posting below and the model will estimate "
    "the probability that it is **fraudulent**."
)

with st.form("job_posting_form"):
    st.subheader("Job details")
    title = st.text_input("Job title", placeholder="e.g. Marketing Intern")
    industry = st.text_input("Industry", placeholder="e.g. Marketing and Advertising")
    function = st.text_input("Function", placeholder="e.g. Marketing")

    col1, col2, col3 = st.columns(3)
    with col1:
        employment_type = st.selectbox("Employment type", EMPLOYMENT_TYPES)
    with col2:
        required_experience = st.selectbox("Required experience", EXPERIENCE_LEVELS)
    with col3:
        required_education = st.selectbox("Required education", EDUCATION_LEVELS)

    st.subheader("Text content")
    company_profile = st.text_area("Company profile", height=100)
    description = st.text_area("Job description", height=150)
    requirements = st.text_area("Requirements", height=150)
    benefits = st.text_area("Benefits", height=100)

    st.subheader("Flags")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        telecommuting = st.checkbox("Telecommuting allowed")
    with fcol2:
        has_company_logo = st.checkbox("Has company logo", value=True)
    with fcol3:
        has_questions = st.checkbox("Has screening questions")

    submitted = st.form_submit_button("Check for fraud")

if submitted:
    record = {
        "title": title,
        "company_profile": company_profile,
        "description": description,
        "requirements": requirements,
        "benefits": benefits,
        "industry": industry,
        "function": function,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
    }

    if not (title or description):
        st.warning("Please provide at least a job title or description before checking.")
    else:
        with st.spinner("Scoring posting..."):
            try:
                result = predict_fraud(record)[0]
            except FileNotFoundError:
                st.error(
                    "Model artifacts not found. Make sure 'artifacts/count_vectorizer.pkl', "
                    "'artifacts/fraud_detection_model.pkl' and 'artifacts/feature_columns.pkl' "
                    "exist relative to this app (or set the ARTIFACTS_DIR environment variable)."
                )
                st.stop()

        probability = result["fraud_probability"]
        label = result["label"]

        st.subheader("Result")
        if result["is_fraudulent"]:
            st.error(f"⚠️ This posting looks **{label}** ({probability:.1%} fraud probability)")
        else:
            st.success(f"✅ This posting looks **{label}** ({probability:.1%} fraud probability)")

        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "Note: this is a statistical estimate based on historical patterns "
            "and should not be treated as a definitive judgement."
        )

st.divider()
st.caption(
    "Model: tuned XGBoost classifier trained on the Kaggle 'Fake Job Postings' dataset "
    "using bag-of-words text features plus job-metadata flags."
)