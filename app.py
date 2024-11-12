import streamlit as st
import fitz  # PyMuPDF
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math

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

# 4. Extract Key Topics (Most frequent words)
def extract_key_topics(word_count, top_n=5):
    # Get the top N most common words (ignoring stopwords)
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_count = {word: count for word, count in word_count.items() if word not in stopwords}
    # Sort the words by frequency and return the top N
    sorted_topics = sorted(filtered_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_topics[:top_n]

# 5. Summarize References to 'Eligibility'
def summarize_eligibility_references(text):
    # Split the text into sentences
    sentences = text.split('.')
    # Find sentences that contain the word "eligibility"
    eligibility_sentences = [sentence.strip() for sentence in sentences if "eligibility" in sentence.lower()]
    # Join those sentences into a summary
    summary = " ".join(eligibility_sentences)
    return summary

# 6. Text Summarization (TextRank-like algorithm) - Without numpy
def summarize_text(text, top_n=5):
    # Split the text into sentences
    sentences = text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Cosine Similarity Calculation (manual, no numpy)
    cosine_sim = cosine_similarity_manual(tfidf_matrix)

    # Build the graph based on sentence similarity
    graph = nx.Graph()
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if cosine_sim[i][j] > 0.2:  # Threshold for considering a connection
                graph.add_edge(i, j, weight=cosine_sim[i][j])

    # Rank the sentences using the graph (PageRank)
    scores = nx.pagerank(graph)

    # Get the top N ranked sentences
    ranked_sentences = sorted(((score, i) for i, score in scores.items()), reverse=True)
    top_sentences = [sentences[i] for _, i in ranked_sentences[:top_n]]
    
    return ' '.join(top_sentences)

def cosine_similarity_manual(tfidf_matrix):
    # Manually compute cosine similarity between sentences
    num_sentences = tfidf_matrix.shape[0]
    cosine_sim = [[0 for _ in range(num_sentences)] for _ in range(num_sentences)]

    for i in range(num_sentences):
        for j in range(i, num_sentences):
            dot_product = sum(tfidf_matrix[i, k] * tfidf_matrix[j, k] for k in range(tfidf_matrix.shape[1]))
            norm_i = math.sqrt(sum(tfidf_matrix[i, k]**2 for k in range(tfidf_matrix.shape[1])))
            norm_j = math.sqrt(sum(tfidf_matrix[j, k]**2 for k in range(tfidf_matrix.shape[1])))
            cosine_sim[i][j] = dot_product / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
            cosine_sim[j][i] = cosine_sim[i][j]  # Symmetric

    return cosine_sim

# 7. Generate Word Embedding Plot - Without numpy
def plot_word_embeddings(text):
    # Tokenize the text
    tokens = tokenize(text)
    
    # Remove stopwords
    stopwords = set(["the", "and", "is", "to", "in", "of", "a", "an", "for", "on", "with", "as", "it", "at", "by", "that", "from", "this", "was", "were", "are", "be", "been", "being"])
    filtered_tokens = [word for word in tokens if word not in stopwords]
    
    # Generate TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_tokens)
    
    # Compute cosine similarity between each pair of words (manual calculation)
    cosine_sim_matrix = cosine_similarity_manual(tfidf_matrix)
    
    # Build the graph for word connections
    G = nx.Graph()
    words = tfidf_vectorizer.get_feature_names_out()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if cosine_sim_matrix[i][j] > 0.5:  # Threshold for visualizing strong relationships
                G.add_edge(words[i], words[j], weight=cosine_sim_matrix[i][j])
    
    # Plot the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    plt.title("Word Embeddings and Their Connections")
    plt.axis('off')
    st.pyplot(plt)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the uploaded file as a binary stream
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Main Streamlit app interface
st.title("PDF Text Extractor and Analysis")
st.write("Upload a PDF file to extract its text, clean it, and analyze key topics, eligibility references, summarize text, and visualize word embeddings.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the uploaded PDF file name
    st.write(f"File: {uploaded_file.name}")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)

    # Tokenize the cleaned text
    tokens = tokenize(cleaned_text)

    # Vectorize the tokens (Bag of Words model)
    word_count = vectorize(tokens)

    # Extract key topics
    key_topics = extract_key_topics(word_count)

    # Summarize references to 'eligibility'
    eligibility_summary = summarize_eligibility_references(extracted_text)

    # Summarize the entire document using TextRank-like summarization
    summarized_text = summarize_text(extracted_text)

    # Display the results
    st.subheader("Extracted Text")
    st.text_area("Text from PDF", extracted_text, height=300)

    st.subheader("Key Topics")
    if key_topics:
        st.write(key_topics)
    else:
        st.write("No key topics could be identified.")

    st.subheader("Eligibility Summary")
    if eligibility_summary:
        st.write(eligibility_summary)
    else:
        st.write("No references to 'eligibility' found.")
    
    st.subheader("Summarized Text")
    st.write(summarized_text)

    # Plot Word Embeddings
    plot_word_embeddings(extracted_text)
