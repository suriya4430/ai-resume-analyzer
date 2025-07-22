import streamlit as st
import fitz  # PyMuPDF
import openai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- SETTINGS --------------------
openai.api_key = "sk-xxxx"  # 🔴 Replace this with your OpenAI API key

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to get a match score!")

# -------------------- FUNCTIONS --------------------
def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def calculate_similarity(resume_text, jd_text):
    documents = [resume_text, jd_text]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

def get_gpt_feedback(resume_text, jd_text):
    prompt = f"""
You are a resume coach. Analyze the following resume against the job description and give improvement suggestions.

Resume:
{resume_text}

Job Description:
{jd_text}

Suggestions:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error: {str(e)}"

# -------------------- MAIN APP --------------------
resume_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste the job description here", height=200)

if resume_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(job_description)
        score = calculate_similarity(clean_resume, clean_jd)
        feedback = get_gpt_feedback(resume_text, job_description)

    st.markdown(f"### ✅ Resume Match Score: **{score}%**")
    
    if score > 75:
        st.success("Great match! Your resume aligns well with the job.")
    elif score > 50:
        st.warning("Decent match. You can improve it further with suggestions below.")
    else:
        st.error("Low match. Consider improving your resume with these suggestions.")

    st.markdown("### 🧠 GPT Resume Suggestions:")
    st.info(feedback)
