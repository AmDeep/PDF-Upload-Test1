import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import PyPDF2
import re

# Load Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to get a summary of the PDF content using Hugging Face BART model
def get_summary(text):
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Function to extract eligibility-related information using Named Entity Recognition (NER)
def get_eligibility_info(text):
    # Extract entities related to eligibility (we'll look for the word 'eligibility' and nearby entities)
    entities = ner_pipeline(text)
    eligibility_info = []
    
    # Search for eligibility-related terms
    for entity in entities:
        if 'eligibility' in entity['word'].lower():
            eligibility_info.append(entity)
    
    # If no eligibility-related entities found, look for keywords
    if not eligibility_info:
        eligibility_info = [entity for entity in entities if re.search(r'\b(eligibility|eligible|eligibility criteria|eligible)\b', entity['word'], re.I)]
    
    # Return the filtered results
    return eligibility_info if eligibility_info else "No specific eligibility information found."

# Streamlit app UI
st.title("PDF Document Summary and Eligibility Extractor")
st.write("Upload a PDF document and generate a summary along with information related to 'eligibility'.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    # Display a preview of the extracted text (first 500 characters)
    st.write("Extracted text preview:")
    st.write(pdf_text[:500])

    # If the document is too long, chunk it (optional, but recommended)
    if len(pdf_text.split()) > 1000:
        st.write("Document is large. It may be split into smaller parts for processing.")

    # Display buttons for generating summary and eligibility information
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            summary = get_summary(pdf_text)
            st.write("Summary:")
            st.write(summary)

    if st.button("Extract Eligibility Information"):
        with st.spinner("Extracting eligibility information..."):
            eligibility_info = get_eligibility_info(pdf_text)
            st.write("Eligibility Information:")
            st.write(eligibility_info)
