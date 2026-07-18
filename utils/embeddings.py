from sentence_transformers import SentenceTransformer


def load_embedding_model():
    """
    Loads the embedding model.

    Returns:
        SentenceTransformer
    """

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


def create_embeddings(model, chunks):
    """
    Creates embeddings for document chunks.

    Parameters:
        model
        chunks

    Returns:
        embeddings
    """

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    return embeddings