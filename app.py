import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import nltk
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora
from nltk.probability import FreqDist
import io

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(io.BytesIO(pdf_file.read()))  # Read PDF from stream
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to clean and preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Function to plot word frequency
def plot_word_frequency(tokens):
    fdist = FreqDist(tokens)
    common_words = fdist.most_common(20)
    words, counts = zip(*common_words)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(words, counts)
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_title("Top 20 Words in Document")
    st.pyplot(fig)

# Function to create a word cloud
def create_word_cloud(tokens):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(tokens))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud of Document")
    st.pyplot(fig)

# Function to perform topic modeling using LDA
def perform_topic_modeling(text, num_topics=3):
    tokens = preprocess_text(text)
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    
    lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    
    st.subheader("Topic Modeling Results:")
    for topic in topics:
        st.write(topic)

    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    st.write(vis)

# Function to find sentences related to the word 'eligibility'
def extract_eligibility_info(text):
    sentences = sent_tokenize(text)
    eligibility_sentences = [sentence for sentence in sentences if 'eligibility' in sentence.lower()]
    return " ".join(eligibility_sentences)

# Streamlit UI
st.title('PDF Document Analysis')
st.markdown('Upload a PDF to analyze word frequency, topics, and eligibility-related information.')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Display the raw text (optional)
    if st.checkbox("Show raw text"):
        st.text_area("Raw text", text, height=300)
    
    # Preprocess and analyze text
    tokens = preprocess_text(text)
    
    # Plot word frequency
    st.subheader("Word Frequency")
    plot_word_frequency(tokens)
    
    # Create a word cloud
    st.subheader("Word Cloud")
    create_word_cloud(tokens)
    
    # Perform topic modeling
    st.subheader("Topic Modeling")
    perform_topic_modeling(text)
    
    # Extract eligibility-related information
    st.subheader("Eligibility Information")
    eligibility_info = extract_eligibility_info(text)
    if eligibility_info:
        st.write(eligibility_info)
    else:
        st.write("No information found regarding 'eligibility'.")
