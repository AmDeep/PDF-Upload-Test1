import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
from io import StringIO

# 1. Data Cleaning
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters except for spaces and alphanumeric characters (e.g., remove periods)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Tokenization
def tokenize(text):
    return text.split()

# 3. Extract Text and Tables from PDF
def extract_text_and_tables_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    page_info = {}
    tables = []

    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        
        # Extract text using standard text extraction
        page_text = page.get_text("text")  # Standard text extraction
        text += page_text

        # Try to extract tables from the page using layout analysis (this works well for tables)
        page_tables = page.get_text("table")  # This returns tables as structured text
        if page_tables:
            tables.append(page_tables)
        
        # Record page number for each sentence
        sentences = page_text.split('.')
        for sentence in sentences:
            page_info[sentence.strip()] = page_num  # Store the page number for each sentence
    
    return text, page_info, tables

# 4. Summarize Mentions of the User-Input Term
def summarize_mentions(text, term, page_info):
    term = term.lower()
    sentences = text.split('.')
    summary_data = []

    # Find all sentences that mention the term
    for sentence in sentences:
        sentence = sentence.strip()
        if term in sentence:
            page_num = page_info.get(sentence, "Unknown page")
            summary_data.append(f"Page {page_num}: {sentence}")
    
    # Return a concise summary of mentions, including single word mentions
    if summary_data:
        return "\n".join(summary_data)
    else:
        return f"No mentions of '{term}' found in the document."

# 5. Generate Contextual Questions
def generate_dynamic_questions(text, term, page_info):
    term = term.lower()
    
    # Extract contextual relationships
    context_data = extract_contextual_relationships(text, term, page_info)
    
    # Generate dynamic questions based on context
    questions = []
    if context_data:
        questions.append(f"What is mentioned about '{term}' in the document?")
        questions.append(f"Can you provide examples of '{term}' being discussed in the document?")
        
        # Check for policy, rules, or definitions
        if any("requirement" in sentence.lower() for sentence in [entry['sentence'] for entry in context_data]):
            questions.append(f"What requirements or rules are associated with '{term}'?")
        
        if any("defined" in sentence.lower() for sentence in [entry['sentence'] for entry in context_data]):
            questions.append(f"How is '{term}' defined in the document?")
        
        # Comparative questions if term appears in multiple contexts
        if len(context_data) > 1:
            questions.append(f"How does the discussion of '{term}' differ in various sections of the document?")
    
    return questions

# 6. Extract Contextual Relationships
def extract_contextual_relationships(text, term, page_info):
    """
    Extract and find sentences or mentions of the term and their page number.
    """
    term = term.lower()
    sentences = text.split('.')
    context_data = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if term in sentence:
            page_num = page_info.get(sentence, "Unknown page")
            relevant_words = [word for word in sentence.split() if word != term]
            context_data.append({
                "sentence": sentence,
                "related_terms": relevant_words,
                "page_number": page_num
            })
    
    # If no context found, look for single word occurrences
    if not context_data:
        for page_num, page_text in page_info.items():
            if term in page_text:
                context_data.append({
                    "sentence": term,  # Single term mention
                    "related_terms": [],
                    "page_number": page_num
                })
    
    return context_data

# 7. Display Extracted Data (Including First 50 Pages if PDF is Large)
def display_first_50_pages(extracted_text, page_info):
    pages = extracted_text.split('\n')
    page_count = len(pages)
    if page_count > 50:
        # Display only the first 50 pages of the document
        pages = pages[:50]
        st.write("\n".join(pages))
    else:
        st.write(extracted_text)

# Main Streamlit app interface
st.title("PDF Text Extractor and Contextual Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze content based on a custom term.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text and tables from the uploaded PDF
    extracted_text, page_info, tables = extract_text_and_tables_from_pdf(uploaded_file)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Input for custom term (e.g., "eligibility")
    custom_term = st.text_input("Enter a term to summarize references (e.g., 'eligibility')", "eligibility")

    # Generate dynamic question prompts
    dynamic_questions = generate_dynamic_questions(cleaned_text, custom_term, page_info)

    # Display dynamic questions
    st.subheader("Sample Questions Based on Your Text")
    for question in dynamic_questions:
        if st.button(question):
            # Generate and display a response to the clicked question
            response = generate_response_to_question(extracted_text, question, custom_term, page_info)
            st.write(f"Response: {response}")

    # Extract and display all contextual mentions of the custom term in the document
    context_data = extract_contextual_relationships(extracted_text, custom_term, page_info)
    st.subheader(f"Contextual Mentions of '{custom_term.capitalize()}'")
    if context_data:
        for entry in context_data:
            st.write(f"Page {entry['page_number']}: {entry['sentence']}")
            st.write(f"Related Terms: {', '.join(entry['related_terms']) if entry['related_terms'] else 'none'}")
    else:
        st.write(f"No contextual mentions of '{custom_term}' found.")
    
    # Generate and display a summary of mentions for the custom term
    st.subheader(f"Summary of Mentions of '{custom_term.capitalize()}'")
    summary = summarize_mentions(extracted_text, custom_term, page_info)
    st.write(summary)

    # Show extracted raw text from the PDF (first 50 pages if document has more than 50 pages)
    st.subheader("Extracted Text from Document")
    display_first_50_pages(extracted_text, page_info)
    
    # Show extracted tables (if any)
    if tables:
        st.subheader("Extracted Tables")
        for idx, table in enumerate(tables):
            st.write(f"Table {idx + 1}:")
            st.write(table)
