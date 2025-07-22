import streamlit as st
import fitz  # PyMuPDF
import openai
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API Key
openai.api_key = "sk-xxxx"  # Replace this with your actual key

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Function to calculate similarity
def calculate_similarity(resume_text, job_desc):
    documents = [resume_text, job_desc]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(score[0][0]) * 100, 2)

# Function to get GPT suggestions
def get_gpt_feedback(resume_text, job_desc):
    prompt = f"""You are a resume improvement assistant.
    Compare the following resume and job description, and suggest improvements to better align the resume with the job role.

    Resume:
    {resume_text}

    Job Description:
    {job_desc}

    Give clear, bullet-point improvement suggestions:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer (GenAI Project)")
st.markdown("Upload your resume and paste a job description to get a match score and GPT-powered improvement suggestions.")

resume_file = st.file_uploader("📎 Upload Resume (PDF only)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description")

if st.button("Analyze") and resume_file and job_description:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        resume_text_clean = clean_text(resume_text)
        job_desc_clean = clean_text(job_description)

        score = calculate_similarity(resume_text_clean, job_desc_clean)
        suggestions = get_gpt_feedback(resume_text_clean, job_desc_clean)

        st.subheader("✅ Match Score:")
        st.success(f"Your resume matches the job description by {score}%")

        st.subheader("💡 GPT Improvement Suggestions:")
        st.markdown(suggestions)
