from utils.pdf_reader import extract_text_from_pdfs
from utils.chunking import create_chunks
from utils.embeddings import (
    load_embedding_model,
    create_embeddings
)
import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:

    st.title("📚 AI Assistant")

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📄 Process Documents"):

      if uploaded_files:

          with st.spinner("Reading PDF documents..."):

              raw_text = extract_text_from_pdfs(uploaded_files)
              chunks = create_chunks(raw_text)
              model = load_embedding_model()

              embeddings = create_embeddings(
                model,
                chunks
              )
              st.write(f"Embedding Dimension: {len(embeddings[0])}")

              st.write(f"Total Embeddings: {len(embeddings)}")

          st.success("PDFs processed successfully!")

          st.subheader("Document Statistics")

          st.write(f"Total Characters: {len(raw_text)}")

          st.write(f"Total Chunks: {len(chunks)}")

          st.markdown("---")

          st.subheader("Chunk Preview")

          for index, chunk in enumerate(chunks):

            with st.expander(f"Chunk {index + 1}"):

                st.write(chunk)

                st.warning("Please upload at least one PDF.")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# ---------------------------
# Main Page
# ---------------------------

st.title("📚 AI Research Assistant")

st.write(
    "Upload your PDF documents and ask questions using Retrieval-Augmented Generation (RAG)."
)

st.markdown("---")

# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        response = (
            "RAG is not connected yet. We'll build that in the next phases."
        )

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )