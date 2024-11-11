import streamlit as st
from PyPDF2 import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text, num_sentences=3):
    """Summarize text using TextRank or TF-IDF method."""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # If the document is short, just return the sentences as is
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Use TF-IDF to compute sentence similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Rank sentences based on their similarity to the others
    sentence_scores = np.sum(cosine_sim, axis=1)
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-num_sentences:][::-1]]

    return " ".join(ranked_sentences)

def main():
    st.title("PDF Summary Generator (TextRank + TF-IDF)")

    st.write("Upload a PDF document to get a summary of its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            st.subheader("Extracted Text")
            st.write(text[:1000])  # Display the first 1000 characters

            # Generate the summary using TF-IDF / TextRank
            st.subheader("Generated Summary")
            summary = summarize_text(text)
            st.write(summary)
        else:
            st.warning("No text could be extracted from this PDF.")

if __name__ == "__main__":
    main()
