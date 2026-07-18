from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(text):
    """
    Splits text into overlapping chunks.

    Parameters:
        text (str): Extracted PDF text

    Returns:
        list: List of text chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(text)

    return chunks