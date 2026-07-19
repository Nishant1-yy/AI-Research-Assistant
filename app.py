from utils.vector_store import (
    create_faiss_index,
    save_index,
    load_index,
    save_chunks,
    load_chunks,
)
from utils.pdf_reader import extract_text_from_pdfs
from utils.chunking import create_chunks
from utils.embeddings import load_embedding_model, create_embeddings
from utils.retriever import retrieve_chunks
import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

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
        "Upload PDF Files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("📄 Process Documents"):

        if uploaded_files:

            with st.spinner("Reading PDF documents..."):

                st.write("Step 1")
                raw_text = extract_text_from_pdfs(uploaded_files)

                st.write("Step 2")
                chunks = create_chunks(raw_text)

                st.write("Step 3")
                model = load_embedding_model()

                st.write("Step 4")
                embeddings = create_embeddings(model, chunks)

                st.write("Step 5")
                index = create_faiss_index(embeddings)

                st.write("Step 6")
                save_index(index)
                save_chunks(chunks)

                st.write("Done")
                st.write(f"Vectors Stored: {index.ntotal}")

                st.write(f"Embedding Dimension: {len(embeddings[0])}")

                st.write(f"Total Embeddings: {len(embeddings)}")

            st.success("PDFs processed successfully!")

            st.subheader("Document Statistics")

            st.write(f"Total Characters: {len(raw_text)}")

            st.write(f"Total Chunks: {len(chunks)}")

            st.markdown("---")

            st.subheader("Chunk Preview")

            for idx, chunk in enumerate(chunks):

                with st.expander(f"Chunk {idx + 1}"):

                    st.write(chunk)

        else:
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

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            model = load_embedding_model()

            index = load_index()

            chunks = load_chunks()

            retrieved = retrieve_chunks(question, model, index, chunks, top_k=3)

            response = ""

            for i, chunk in enumerate(retrieved):

                response += f"### Retrieved Chunk {i+1}\n\n"

                response += chunk

                response += "\n\n------------------\n\n"

        except Exception as e:

            response = str(e)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
