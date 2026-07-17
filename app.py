import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚"
)

st.title("📚 AI Research Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.success("PDF Loaded Successfully!")

    st.subheader("Extracted Text")

    st.write(text[:3000])   # Show first 3000 characters