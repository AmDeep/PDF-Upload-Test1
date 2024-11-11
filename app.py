import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate vector embeddings for content
def generate_embeddings(text, model):
    sentences = text.split('.')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings, sentences

# Function to search for 'eligibility' related sentences
def search_eligibility_sentences(sentences, embeddings, query, faiss_index):
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, I = faiss_index.search(query_embedding.cpu().numpy(), k=5)
    results = []
    for idx in I[0]:
        results.append((sentences[idx], D[0][idx]))
    return results

# Streamlit UI
st.title("PDF Document Processing with Vector Embedding")
st.write("Upload a PDF document to generate vector embeddings and analyze content related to 'eligibility'.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Extract the content from the uploaded PDF
    with st.spinner("Extracting text from the PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    # Display the extracted text (first 500 characters as a preview)
    st.write("Extracted text preview:")
    st.write(pdf_text[:500])

    # Use a pre-trained SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings, sentences = generate_embeddings(pdf_text, model)

    # Create a FAISS index for fast similarity search
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings.cpu().numpy(), dtype=np.float32))

    # Submit button
    if st.button('Submit and Generate Results'):
        # Search for content related to the word 'eligibility'
        eligibility_results = search_eligibility_sentences(sentences, embeddings, 'eligibility', faiss_index)

        # Display the top results
        if eligibility_results:
            st.write(f"Top results related to 'eligibility':")
            for sentence, score in eligibility_results:
                st.write(f"Score: {score:.4f} - {sentence}")
        else:
            st.write("No content related to 'eligibility' was found.")
