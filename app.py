import streamlit as st
import fitz  # PyMuPDF
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

# 4. Generate Detailed Summary of Text
def generate_detailed_summary(text, top_n=5):
    # Tokenize and vectorize the text
    tokens = tokenize(text)
    word_count = vectorize(tokens)
    
    # Filter out common words (stopwords) from the word count
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_count = {word: count for word, count in word_count.items() if word not in stopwords}
    
    # Sort the words by frequency
    sorted_word_freq = sorted(filtered_count.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N frequent words
    top_words = sorted_word_freq[:top_n]
    
    # Generate a summary of word frequencies and the top N words
    detailed_summary = {
        "Total Words": len(tokens),
        "Unique Words": len(filtered_count),
        "Top Words": top_words
    }
    return detailed_summary

# 5. Summarize References to Custom Term (e.g., "eligibility")
def summarize_term_references(text, term="eligibility"):
    # Split the text into sentences
    sentences = text.split('.')
    # Find sentences that contain the given term
    term_sentences = [sentence.strip() for sentence in sentences if term.lower() in sentence.lower()]
    # Join those sentences into a summary
    summary = " ".join(term_sentences)
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

# Function to find all mentions of a custom term or similar words
def find_all_mentions(text, term="eligibility"):
    # Using a simple word matching method for finding the term and similar words
    term_variants = [term, term + "s", term + "es", term + "ed", term + "ing"]
    mentions = []
    
    # Split the text into sentences and find all occurrences
    sentences = text.split('.')
    for sentence in sentences:
        if any(variant in sentence.lower() for variant in term_variants):
            mentions.append(sentence.strip())
    
    return mentions

# Function to extract key concepts/phrases from text (without spaCy)
def extract_key_concepts(text):
    # Use regular expressions to find noun-like phrases
    noun_like_phrases = re.findall(r'\b(?:[a-zA-Z]+(?: [a-zA-Z]+){1,3})\b', text)
    
    # Remove common stopwords from the noun-like phrases
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_phrases = [phrase for phrase in noun_like_phrases if phrase not in stopwords]
    
    # Remove duplicates and sort the phrases for clarity
    filtered_phrases = list(set(filtered_phrases))
    filtered_phrases.sort()
    
    return filtered_phrases

# Function to generate dynamic question prompts based on extracted concepts
def generate_dynamic_questions(concepts):
    questions = []
    
    for concept in concepts:
        questions.append(f"What is mentioned about '{concept}'?")
        questions.append(f"How is '{concept}' defined?")
        questions.append(f"Where does the document mention '{concept}'?")
        questions.append(f"Can you provide examples of '{concept}' from the document?")
    
    return questions

# Main Streamlit app interface
st.title("PDF Text Extractor and Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze content based on key terms.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the uploaded PDF file name
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Generate a detailed summary
    detailed_summary = generate_detailed_summary(cleaned_text)

    # Input for custom term (e.g., "eligibility")
    custom_term = st.text_input("Enter a term to summarize references (e.g., 'eligibility')", "eligibility")

    # Summarize references to the custom term
    term_summary = summarize_term_references(extracted_text, term=custom_term)

    # Extract key concepts/phrases from the text for dynamic question generation
    key_concepts = extract_key_concepts(cleaned_text)

    # Generate dynamic question prompts
    dynamic_questions = generate_dynamic_questions(key_concepts)

    # Display sample question prompts based on the extracted concepts
    st.subheader("Sample Questions Based on Your Text")
    for question in dynamic_questions:
        st.button(question)

    # Display all mentions of the custom term in the document
    mentions = find_all_mentions(extracted_text, term=custom_term)
    
    st.subheader(f"All Mentions of '{custom_term.capitalize()}'")
    if mentions:
        st.write("\n".join(mentions))
    else:
        st.write(f"No mentions of '{custom_term}' found.")
    
    # Display the results
    st.subheader("Extracted Text")
    st.text_area("Text from PDF", extracted_text, height=300)

    st.subheader("Detailed Summary of Text")
    if detailed_summary:
        st.write(f"Total Words: {detailed_summary['Total Words']}")
        st.write(f"Unique Words: {detailed_summary['Unique Words']}")
        st.write("Top Words and their Frequencies:")
        for word, freq in detailed_summary['Top Words']:
            st.write(f"{word}: {freq}")
    else:
        st.write("No detailed summary could be generated.")

    st.subheader(f"Summary of References to '{custom_term.capitalize()}'")
    if term_summary:
        st.write(term_summary)
    else:
        st.write(f"No references to '{custom_term}' found.")
