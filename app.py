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
    return text.split()

# 3. Vectorization (Simple Bag of Words)
def vectorize(tokens):
    return Counter(tokens)

# 4. Extract Contextual Relationships
def extract_contextual_relationships(text, term):
    """
    Analyze the contextual relationships between the user input term
    and other words in the document to generate contextually rich data.
    """
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
    
    return context_data

# 5. Summarize Mentions of the User-Input Text
def summarize_mentions(text, term):
    """
    Summarize all mentions of the user-input term in relation to the document.
    """
    term = term.lower()
    sentences = text.split('.')
    summary_data = []

    # Find all sentences that mention the term
    for sentence in sentences:
        sentence = sentence.strip()
        if term in sentence:
            summary_data.append(sentence)
    
    # Return a concise summary of mentions
    if summary_data:
        return "\n".join(summary_data)
    else:
        return f"No mentions of '{term}' found in the document."

# Function to generate dynamic question prompts based on the extracted term
def generate_dynamic_questions(text, term):
    term = term.lower()
    
    # Extract contextual relationships
    context_data = extract_contextual_relationships(text, term)
    
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

# Function to generate contextual response to a question
def generate_response_to_question(text, question, term):
    term = term.lower()
    
    # Extract contextual relationships
    context_data = extract_contextual_relationships(text, term)
    
    # Identify question type and generate smart, context-aware responses
    if "about" in question or "what" in question.lower():
        if context_data:
            response = f"The document discusses '{term}' in various contexts: "
            for entry in context_data:
                response += f"\n- In the sentence: '{entry['sentence']}', related terms are {', '.join(entry['related_terms'])}."
            return response
        else:
            return f"'{term}' is only briefly mentioned or not fully explored in the document."

    elif "examples" in question.lower():
        examples = [entry['sentence'] for entry in context_data if "example" in entry['sentence'].lower()]
        if examples:
            return f"Here is an example of '{term}' in the document: {examples[0]}"
        else:
            return f"No clear examples of '{term}' were found in the document."

    elif "requirements" in question.lower() or "rules" in question.lower():
        requirements = [entry['sentence'] for entry in context_data if "requirement" in entry['sentence'].lower()]
        if requirements:
            return f"'{term}' is associated with specific eligibility requirements, such as {requirements[0]}"
        else:
            return f"No specific eligibility requirements related to '{term}' were found in the document."

    elif "defined" in question.lower():
        definitions = [entry['sentence'] for entry in context_data if "defined" in entry['sentence'].lower()]
        if definitions:
            return f"'{term}' is defined in the document as: {definitions[0]}"
        else:
            return f"'{term}' is not explicitly defined in the document."

    elif "different" in question.lower() and len(context_data) > 1:
        return f"Across different sections, '{term}' is discussed from various perspectives, such as eligibility conditions, examples of qualifying factors, and eligibility rules."

    else:
        return f"The document offers a detailed exploration of '{term}', providing insight into its significance in relation to other policy terms."

# Function to extract text from PDF (including tables)
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    
    # Iterate through each page
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        
        # Extract text using layout analysis (this helps to capture table-like structures)
        text += page.get_text("text")  # Standard text extraction
        # If needed, use 'html' or 'xml' mode to preserve structure (useful for tables)
        # text += page.get_text("html")  # This might help with table extraction
    
    return text

# Main Streamlit app interface
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
    else:
        st.write(f"No contextual mentions of '{custom_term}' found.")
    
    # Generate and display a summary of mentions for the custom term
    st.subheader(f"Summary of Mentions of '{custom_term.capitalize()}'")
    summary = summarize_mentions(extracted_text, custom_term)
    st.write(summary)
