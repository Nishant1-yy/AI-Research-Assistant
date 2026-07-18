# AI Research Assistant (RAG)

An AI-powered Research Assistant that allows users to upload PDF documents and ask questions using Retrieval-Augmented Generation (RAG).

## Features

- Upload multiple PDF files
- Extract text from PDFs
- Intelligent text chunking
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Streamlit web interface
- Local LLM support using Ollama (coming soon)

## Tech Stack

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- LangChain
- Ollama

## Project Structure

AI-Research-Assistant/
├── app.py
├── utils/
├── vector_store/
├── requirements.txt
└── README.md

## Installation

```bash
git clone <repo-url>
cd AI-Research-Assistant

python -m venv venv

# Activate the virtual environment

pip install -r requirements.txt

streamlit run app.py