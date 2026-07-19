import numpy as np


def retrieve_chunks(
    question,
    model,
    index,
    chunks,
    top_k=3
):
    """
    Retrieves the most relevant chunks.
    """

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        retrieved_chunks.append(
            chunks[idx]
        )

    return retrieved_chunks