from pypdf import PdfReader


def extract_text_from_pdfs(uploaded_files):
    """
    Extracts text from multiple uploaded PDF files.

    Parameters:
        uploaded_files (list): List of uploaded PDF files.

    Returns:
        str: Combined text from all PDFs.
    """

    combined_text = ""

    for pdf in uploaded_files:

        reader = PdfReader(pdf)

        for page in reader.pages:

            text = page.extract_text()

            if text:
                combined_text += text + "\n"

    return combined_text