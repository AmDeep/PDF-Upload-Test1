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

# Function to generate dynamic question prompts based on the extracted term
def generate_dynamic_questions(text, term):
    # Normalize the text and term to lowercase
    term = term.lower()
    text = text.lower()
    
    # Split the text into sentences
    sentences = text.split('.')
    
    # Find all sentences that contain the term or its variants
    relevant_sentences = [sentence.strip() for sentence in sentences if term in sentence]
    
    # Generate up to 5 dynamic questions based on the term's context in the document
    questions = []
    
    # Basic question structure
    questions.append(f"What is mentioned about '{term}' in the document?")
    
    if relevant_sentences:
        questions.append(f"What are the key examples or details provided about '{term}'?")
        questions.append(f"Where is '{term}' discussed in the document?")
        
        # Check if there are multiple references to create a comparative question
        if len(relevant_sentences) > 1:
            questions.append(f"How are the mentions of '{term}' different across the document?")
        
        questions.append(f"How is '{term}' defined or explained?")
    
    # Limit the number of questions to 5
    return questions[:5]

# Function to generate contextual response to a question
def generate_response_to_question(text, question, term):
    # Normalize the text and term to lowercase
    term = term.lower()
    text = text.lower()
    
    # Split the text into sentences
    sentences = text.split('.')
    
    # Find sentences related to the question
    relevant_sentences = [sentence.strip() for sentence in sentences if term in sentence]
    
    # Generate a response based on the question
    if "about" in question or "what" in question.lower():
        return "This document discusses the topic of '{}' in several places. It explains it primarily in terms of [contextual explanation].".format(term)
    elif "examples" in question.lower():
        if relevant_sentences:
            return "One example of '{}' in this document is [example extracted]."
        else:
            return "No specific examples were provided in the document regarding '{}'."
    elif "discussed" in question.lower():
        return "The topic of '{}' is discussed throughout the document, especially in sections [related sections]."
    elif "defined" in question.lower():
        return "'{}' is defined as [definition extracted].".format(term)
    elif "different" in question.lower() and len(relevant_sentences) > 1:
        return "Mentions of '{}' vary across the document, with different sections focusing on different aspects of the term.".format(term)
    else:
        return "This document contains detailed references to '{}', providing various perspectives.".format(term)

# Main Streamlit app interface
st.title("PDF Text Extractor and Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze content based on a custom term.")

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

    # Generate dynamic question prompts
    dynamic_questions = generate_dynamic_questions(cleaned_text, custom_term)

    # Display sample question prompts based on the extracted term
    st.subheader("Sample Questions Based on Your Text")
    for question in dynamic_questions:
        if st.button(question):
            # Generate and display a response to the clicked question
            response = generate_response_to_question(extracted_text, question, custom_term)
            st.write(f"Response: {response}")

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
