import streamlit as st
import fitz  # PyMuPDF
import re
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

# 3. Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    page_info = {}

    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        
        # Extract text using standard text extraction
        page_text = page.get_text("text")  # Standard text extraction
        text += page_text

        # Record page number for each sentence
        sentences = page_text.split('.')
        for sentence in sentences:
            page_info[sentence.strip()] = page_num  # Store the page number for each sentence
    
    return text, page_info

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
            # Bold and underline the term for better visibility
            highlighted_sentence = sentence.replace(term, f"**_{term}_**")
            summary_data.append(f"Page {page_num}: {highlighted_sentence}")
    
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
        for sentence, page_num in page_info.items():
            if term in sentence:
                context_data.append({
                    "sentence": term,  # Single term mention
                    "related_terms": [],
                    "page_number": page_num
                })
    
    return context_data

# 7. Display Pages where Term is Found
def display_pages_with_term(page_info, term):
    term = term.lower()
    pages_found = set()

    # Iterate through page_info to find where the term appears
    for sentence, page_num in page_info.items():
        if term in sentence.lower():  # Check if the term is in the sentence
            pages_found.add(page_num + 1)  # Page number is 1-based in the output
    
    if pages_found:
        return f"The term '{term}' was found on the following pages: " + ", ".join(map(str, sorted(pages_found)))
    else:
        return f"No mentions of '{term}' found in the document."

# Main Streamlit app interface
st.title("PDF Text Extractor and Contextual Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze content based on a custom term.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text, page_info = extract_text_from_pdf(uploaded_file)

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
            # Underline and bold the term in the sentence
            highlighted_sentence = entry['sentence'].replace(custom_term, f"**_{custom_term}_**")
            st.write(f"Page {entry['page_number']}: {highlighted_sentence}")
            st.write(f"Related Terms: {', '.join(entry['related_terms']) if entry['related_terms'] else 'none'}")
    else:
        st.write(f"No contextual mentions of '{custom_term}' found.")
    
    # Generate and display a summary of mentions for the custom term
    st.subheader(f"Summary of Mentions of '{custom_term.capitalize()}'")
    summary = summarize_mentions(extracted_text, custom_term, page_info)
    st.write(summary)

    # Show list of pages where the term is found
    st.subheader(f"Pages with Mentions of '{custom_term.capitalize()}'")
    pages_with_term = display_pages_with_term(page_info, custom_term)
    st.write(pages_with_term)
