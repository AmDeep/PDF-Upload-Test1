import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import re
from collections import Counter

# 1. Data Cleaning
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Tokenization
def tokenize(text):
    # Split the text into words based on spaces
    tokens = text.split()
    return tokens

# 3. Vectorization (Simple Bag of Words)
def vectorize(tokens):
    # Create a dictionary with the frequency of each token
    word_count = Counter(tokens)
    return word_count

# 4. Extract Key Topics (Most frequent words)
def extract_key_topics(word_count, top_n=5):
    # Get the top N most common words (ignoring stopwords)
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_count = {word: count for word, count in word_count.items() if word not in stopwords}
    # Sort the words by frequency and return the top N
    sorted_topics = sorted(filtered_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:top_n]

# 5. Summarize References to 'Eligibility'
def summarize_eligibility_references(text):
    # Split the text into sentences
    sentences = text.split('.')
    # Find sentences that contain the word "eligibility"
    eligibility_sentences = [sentence.strip() for sentence in sentences if "eligibility" in sentence.lower()]
    # Join those sentences into a summary
    summary = " ".join(eligibility_sentences)
    return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the uploaded file as a binary stream
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Main Streamlit app interface
st.title("PDF Text Extractor and Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze key topics and references to 'eligibility'.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the uploaded PDF file name
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Tokenize the cleaned text
    tokens = tokenize(cleaned_text)

    # Vectorize the tokens (Bag of Words model)
    word_count = vectorize(tokens)

    # Extract key topics
    key_topics = extract_key_topics(word_count)

    # Summarize references to 'eligibility'
    eligibility_summary = summarize_eligibility_references(extracted_text)

    # Display the results
    st.subheader("Extracted Text")
    st.text_area("Text from PDF", extracted_text, height=300)

    st.subheader("Key Topics")
    if key_topics:
        st.write(key_topics)
    else:
        st.write("No key topics could be identified.")

    st.subheader("Eligibility Summary")
    if eligibility_summary:
        st.write(eligibility_summary)
    else:
        st.write("No references to 'eligibility' found.")
