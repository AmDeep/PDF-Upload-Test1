import streamlit as st
from PyPDF2 import PdfReader
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk
nltk.download('punkt')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_summary(text, summarizer_type="TextRank"):
    """Generate a summary using the chosen summarizer."""
    parser = PlaintextParser.from_string(text, PlaintextParser.from_string(text).tokenizer)

    # Choose the summarizer: TextRank or LSA
    if summarizer_type == "TextRank":
        summarizer = TextRankSummarizer()
    elif summarizer_type == "LSA":
        summarizer = LsaSummarizer()
    else:
        summarizer = TextRankSummarizer()

    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])

def main():
    st.title("PDF Summary Generator (Traditional NLP)")

    st.write("Upload a PDF document to get a summary of its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            st.subheader("Extracted Text")
            st.write(text[:1000])  # Display the first 1000 characters

            # Summarize using TextRank or LSA
            st.subheader("Generated Summary (TextRank or LSA)")
            summary = generate_summary(text, summarizer_type="TextRank")  # or use "LSA"
            st.write(summary)
        else:
            st.warning("No text could be extracted from this PDF.")

if __name__ == "__main__":
    main()
