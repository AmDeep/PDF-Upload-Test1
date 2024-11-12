import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the uploaded file as a binary stream
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Streamlit app interface
st.title("PDF Text Extractor")
st.write("Upload a PDF file to extract and display its text.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the uploaded PDF file name
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Display extracted text
    if extracted_text:
        st.subheader("Extracted Text")
        st.text_area("Text from PDF", extracted_text, height=300)
    else:
        st.write("No text could be extracted from this PDF.")
