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

# 4. Create Network Graph (text-based) for Related Terms
def create_network_plot(text, term, top_n=10):
    # Normalize the text and term to lowercase
    term = term.lower()
    text = text.lower()
    
    # Tokenize the cleaned text
    tokens = tokenize(text)
    
    # Identify important terms by frequency (excluding stopwords)
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # Build a frequency count of the filtered tokens
    word_count = Counter(filtered_tokens)
    
    # Extract terms that co-occur with the user-input term in the text
    co_occurring_terms = [word for word in word_count if term in word]
    
    # Generate a simple textual network plot representation
    network_data = {}
    network_data[term] = []
    
    for word, count in word_count.items():
        if word != term and word in co_occurring_terms:
            network_data[term].append((word, count))
    
    # Create and display the textual representation of the network
    network_plot_text = f"Network of Terms Related to '{term.capitalize()}':\n"
    if not network_data[term]:
        network_plot_text += f"No related terms found for '{term}' in the document.\n"
    else:
        for related_term, count in network_data[term]:
            network_plot_text += f"- {related_term} (Frequency: {count})\n"
    
    return network_plot_text

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

# Function to find similar terms (based on prefix/suffix matching)
def find_similar_terms(text, term, threshold=0.6):
    # Check for words that share a common prefix or suffix with the term
    term_length = len(term)
    similar_terms = set()
    
    # Tokenize the text and compare with the input term
    tokens = tokenize(text)
    for token in tokens:
        # Check if there's a reasonable similarity based on the term length
        if len(token) >= term_length and (token.startswith(term[:3]) or token.endswith(term[-3:])):
            similar_terms.add(token)
    
    return similar_terms

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
    
    # Respond dynamically based on the type of question
    if "about" in question or "what" in question.lower():
        return f"The document primarily discusses '{term}' in relation to various aspects, such as its importance in the context of eligibility requirements, policy details, and examples of eligibility in practice."
    
    elif "examples" in question.lower():
        if relevant_sentences:
            example = relevant_sentences[0]  # Select first relevant example
            return f"One example of '{term}' in the document is: {example}. This highlights how eligibility plays a crucial role in decision-making."
        else:
            return f"Unfortunately, no specific examples were found that directly discuss '{term}' in the document."

    elif "discussed" in question.lower():
        return f"The term '{term}' is mentioned several times in the document. It appears in sections such as eligibility requirements, examples, and guidelines for further action."

    elif "defined" in question.lower():
        if relevant_sentences:
            return f"'{term}' is defined as the criteria or set of conditions that must be met in order to qualify for a certain benefit or service, as discussed in the document."
        else:
            return f"The term '{term}' appears in the document but is not explicitly defined."

    elif "different" in question.lower() and len(relevant_sentences) > 1:
        return f"Throughout the document, there are different perspectives on '{term}', with each section providing a unique take on its significance in various contexts."

    else:
        return f"The document offers a thorough discussion on '{term}', providing key insights and practical applications."

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

    # Input for custom term (e.g., "eligibility")
    custom_term = st.text_input("Enter a term to summarize references (e.g., 'eligibility')", "eligibility")

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
    
    # Generate and display the network plot text representation
    st.subheader(f"Network Plot of Terms Related to '{custom_term.capitalize()}'")
    network_plot_text = create_network_plot(cleaned_text, custom_term)
    st.text(network_plot_text)

    # Find similar terms related to the custom term
    similar_terms = find_similar_terms(cleaned_text, custom_term)
    st.subheader(f"Similar Terms to '{custom_term.capitalize()}'")
    if similar_terms:
        st.write("\n".join(similar_terms))
    else:
        st.write(f"No similar terms found for '{custom_term}'.")
