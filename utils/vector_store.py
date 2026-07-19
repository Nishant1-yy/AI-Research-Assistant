import faiss
import numpy as np
import os
import pickle

os.makedirs("vector_store", exist_ok=True)

def save_index(index):

    faiss.write_index(
        index,
        "vector_store/faiss_index.index"
    )

def save_chunks(chunks):
    """
    Saves chunks to disk.
    """

    with open(
        "vector_store/chunks.pkl",
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )


def load_chunks():
    """
    Loads saved chunks.
    """

    with open(
        "vector_store/chunks.pkl",
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return chunks

def load_index():

    index = faiss.read_index(
        "vector_store/faiss_index.index"
    )

    return index


def create_faiss_index(embeddings):
    """
    Creates a FAISS index from embeddings.

    Parameters:
        embeddings (numpy.ndarray)

    Returns:
        FAISS index
    """

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index