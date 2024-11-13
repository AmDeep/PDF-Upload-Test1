import streamlit as st
import fitz  # PyMuPDF
import re
import math
from collections import Counter
from typing import List


# 1. Data Cleaning
def clean_text(text: str) -> str:
    """Cleans the text by removing punctuation, numbers, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text


# 2. Tokenization
def tokenize(text: str) -> List[str]:
    """Tokenizes the text into a list of words."""
    return text.split()


# 3. Vectorization (Simple Bag of Words)
def vectorize(tokens: List[str]) -> Counter:
    """Converts a list of tokens into a frequency count."""
    return Counter(tokens)


# 4. Cosine Similarity
def cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """Calculate cosine similarity between two vectors."""
    intersection = set(vec1) & set(vec2)
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x] ** 2 for x in vec1])
    sum2 = sum([vec2[x] ** 2 for x in vec2])
    
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# 5. Jaccard Similarity
def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    return intersection / union


# 6. Advanced Similarity Function
def advanced_similarity(text: str, term: str) -> List[str]:
    """Return a list of similar terms using cosine similarity and Jaccard similarity."""
    term = term.lower()
    # Tokenize the text and calculate the vector for the input term
    tokens = tokenize(text)
    term_vector = vectorize([term])

    # Find contextually similar terms based on cosine similarity and Jaccard similarity
    similar_terms = []
    for word in set(tokens):  # Iterate through unique words in the document
        if word == term:
            continue
        
        word_vector = vectorize([word])
        cos_sim = cosine_similarity(term_vector, word_vector)
        jaccard_sim = jaccard_similarity(set([term]), set([word]))
        
        # If both similarities are above a certain threshold, consider the term as related
        if cos_sim > 0.2 or jaccard_sim > 0.1:
            similar_terms.append(word)
    
    return similar_terms


# 7. Extract Contextual Relationships (Including Term Mention)
def extract_contextual_relationships(text: str, term: str, page_info=None) -> List[dict]:
    """Extract contextual mentions of the user-entered term."""
    term = term.lower()
    sentences = text.split('.')
    context_data = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if term in sentence:
            words = sentence.split()
            relevant_words = [word for word in words if word not in ["the", "and", "is", "to", "in", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"]]
            
            # Look for related terms based on proximity in the sentence
            related_terms = [word for word in relevant_words if word != term]
            
            context_data.append({
                "sentence": sentence,
                "related_terms": related_terms
            })
    
    # If no context found, look for single word occurrences
    if not context_data and page_info:
        for page_num, page_text in page_info.items():
            if term in page_text:
                context_data.append({
                    "sentence": term,  # Single term mention
                    "related_terms": [],  # No related terms for standalone mention
                    "page_num": page_num
                })
    
    return context_data


# 8. Print Full Lines with the Term (Ordered by Page)
def print_full_lines_with_term(extracted_text: str, term: str, page_info: dict) -> str:
    """Print full lines containing the term, ordered by page."""
    term = term.lower()
    full_lines_with_term = []
    
    # Split the text into lines and track page numbers
    lines = extracted_text.split('\n')
    page_lines = {}
    
    # Collect lines by page
    for line in lines:
        page_num = page_info.get(line.strip(), None)
        if page_num is not None and term in line.lower():
            # Highlight the term by underlining and bolding it
            full_line = line.replace(term, f"**_{term}_**")
            if page_num not in page_lines:
                page_lines[page_num] = []
            page_lines[page_num].append(full_line)
    
    # Order the lines by page number
    sorted_page_numbers = sorted(page_lines.keys())
    for page_num in sorted_page_numbers:
        for line in page_lines[page_num]:
            full_lines_with_term.append(line)
    
    return "\n".join(full_lines_with_term)


# 9. Extract Related Terms from the Document
def extract_related_terms(text: str, term: str) -> List[str]:
    """Extract terms related to or similar to the input term."""
    term = term.lower()
    related_terms = advanced_similarity(text, term)
    return related_terms


# 10. Extract Text from PDF (including table-like structures)
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from the uploaded PDF."""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    
    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")  # Standard text extraction
    
    return text


# Streamlit app interface
st.title("PDF Text Extractor and Contextual Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze content based on a custom term.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)
    
    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Input for custom term (e.g., "eligibility")
    custom_term = st.text_input("Enter a term to summarize references (e.g., 'eligibility')", "eligibility")

    # Generate dynamic question prompts
    dynamic_questions = generate_dynamic_questions(cleaned_text, custom_term)

    # Display dynamic questions
    st.subheader("Sample Questions Based on Your Text")
    for question in dynamic_questions:
        if st.button(question):
            # Generate and display a response to the clicked question
            response = generate_response_to_question(extracted_text, question, custom_term)
            st.write(f"Response: {response}")

    # Extract and display all contextual mentions of the custom term in the document
    context_data = extract_contextual_relationships(extracted_text, custom_term)
    st.subheader(f"Contextual Mentions of '{custom_term.capitalize()}'")
    if context_data:
        for entry in context_data:
            st.write(f"Sentence: {entry['sentence']}")
            st.write(f"Related Terms: {', '.join(entry['related_terms'])}")
    
    # Print full lines with the term (ordered by page number)
    st.subheader("List all the text referencing the user input term")
    full_lines = print_full_lines_with_term(extracted_text, custom_term, context_data)
    st.write(full_lines)

    # Display related terms
    related_terms = extract_related_terms(extracted_text, custom_term)
    st.subheader(f"Terms related to or similar to '{custom_term.capitalize()}'")
    st.write(related_terms)
