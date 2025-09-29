import PyPDF2
import re
import streamlit as st
import google.generativeai as genai
import os

# --- API Key Management ---.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API key not found. Please add GOOGLE_API_KEY to Streamlit secrets or set as an environment variable.")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Initialization ---
try:
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Error initializing Gemini model: {e}")
    st.stop()

def load_and_process_pdf(pdf_file):
    """Loads a PDF, extracts text, and processes it into chunks."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        # Clean the extracted text
        text = text.strip()
        text = re.sub(r'\n+', '\n', text)
        text = text.replace('\f', '')
        # Split the text into chunks
        chunk_size = 1000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return text_chunks
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return None

def answer_question_llm(question, text_chunks):
    """Answers a question based on the provided text chunks using a language model."""
    if not text_chunks:
        return "Error: No text content available from the PDF."
    context = "\n".join(text_chunks)
    try:
        response = gemini_model.generate_content(f"Using the following text, answer the question: {question}\n\nText: {context}")
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {e}"

# --- Streamlit UI ---
st.title("LLM Bot for ISA UK 200 Document")

uploaded_pdf = st.file_uploader("Upload ISA UK 200 PDF", type=["pdf"])
if uploaded_pdf:
    text_chunks = load_and_process_pdf(uploaded_pdf)
    st.success("PDF processed. You can now ask questions about its content.")

    question = st.text_input("Your question:")
    if st.button("Get Answer") and question:
        answer = answer_question_llm(question, text_chunks)
        st.markdown("**Answer:**")
        st.write(answer)
