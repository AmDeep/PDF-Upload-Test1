import streamlit as st
import PyPDF2
import nltk
import spacy
from spacy.cli import download
from sentence_transformers import SentenceTransformer
import numpy as np

# Download the spaCy model if not already available
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to process the text and extract information on 'eligibility'
def extract_eligibility_info(text):
    doc = nlp(text)
    eligibility_sentences = []
    for sent in doc.sents:
        if 'eligibility' in sent.text.lower():
            eligibility_sentences.append(sent.text)
    return eligibility_sentences

# Generate embeddings for the content
def generate_embeddings(text):
    sentences = nltk.sent_tokenize(text)  # Tokenize text into sentences
    embeddings = embedder.encode(sentences)  # Generate embeddings
    return sentences, embeddings

# Main Streamlit app
def main():
    st.title('PDF Document Processing App')
    
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        st.write("Text extracted from the PDF:")
        st.text_area("Extracted Text", text, height=300)
        
        # Button to process the document
        if st.button('Submit'):
            # Extract information about 'eligibility'
            eligibility_info = extract_eligibility_info(text)
            if eligibility_info:
                st.write("Found 'eligibility' related information:")
                for idx, info in enumerate(eligibility_info, 1):
                    st.write(f"{idx}. {info}")
            else:
                st.write("No information found related to 'eligibility'.")
            
            # Generate vector embeddings for the document
            sentences, embeddings = generate_embeddings(text)
            st.write("Document Topics (Embeddings generated for each sentence):")
            
            # Display some embeddings with the associated sentences
            for idx, (sentence, embedding) in enumerate(zip(sentences[:5], embeddings[:5])):  # Show first 5 for brevity
                st.write(f"Sentence {idx + 1}: {sentence}")
                st.write(f"Embedding: {embedding[:10]}...")  # Display first 10 dimensions of the embedding for brevity
            
            st.success("Document processed successfully!")

# Run the app
if __name__ == '__main__':
    main()
