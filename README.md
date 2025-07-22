# 💼 AI Resume Analyzer (GenAI Project)

An AI-powered resume analyzer built using Streamlit and OpenAI that compares your resume with a job description, calculates a match score, and provides GPT-powered improvement suggestions.

## 🚀 Demo

[Click here to try it out!](#)  <!-- Add your Streamlit Cloud link here once deployed -->

---

## 🧠 Features

- 📎 Upload your Resume (PDF)
- 📋 Paste any Job Description
- 📈 Get a Match Score (using Cosine Similarity)
- 🤖 Get Improvement Suggestions (powered by OpenAI GPT)
- 💬 Clean, interactive UI built with Streamlit

---

## 🔧 Tech Stack

- **Python**
- **Streamlit**
- **OpenAI GPT (gpt-3.5-turbo)**
- **PDF Parsing:** PyMuPDF
- **Text Matching:** TF-IDF + Cosine Similarity
- **NLP**: Basic preprocessing
- **Deployment:** Streamlit Cloud (optional)

---

## 🖥️ How It Works

1. Extracts resume text using `PyMuPDF`
2. Cleans and compares resume with JD using `TF-IDF + Cosine Similarity`
3. Sends both to OpenAI's GPT to generate feedback
4. Displays everything in a clean Streamlit interface

---

## 📦 Setup Instructions

1. Clone this repo  
```bash
git clone https://github.com/suriya4430/ai-resume-analyzer.git
cd ai-resume-analyzer


