import streamlit as st
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader

@st.cache_resource
def load_pdf(file):
    loader = PyMuPDFLoader(file)
    documents = loader.load()
    text = " ".join(doc.page_content for doc in documents)
    return text

@st.cache_data
def split_text(text, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

@st.cache_resource
def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embedding_model)

def get_relevant_chunks(vector_store, query, k=5):
    return [result.page_content for result in vector_store.similarity_search(query, k=k)]

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_long_text(summarizer, text, chunk_size=800):
    tokens = text.split()
    summaries = []

    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i:i+chunk_size])
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    final_summary = summarizer(" ".join(summaries), max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return final_summary

st.set_page_config(page_title="Legal Doc Summarizer", layout="wide")
st.title("Legal Document Summarizer and Q&A")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    with st.spinner("Loading and processing document..."):
        document_text = load_pdf(uploaded_file)
        chunks = split_text(document_text)
        vector_store = create_vector_store(chunks)
        summarizer = load_summarizer()

    st.success("Document processed successfully!")

    if st.button("Summarize Document"):
        with st.spinner("Summarizing document..."):
            full_text = " ".join(chunks[:5])  # Adjust if needed
            summary = summarize_long_text(summarizer, full_text)
            st.subheader(" Document Summary")
            st.write(summary)

    st.markdown("---")
    st.subheader(" Ask a Question from the Document")
    user_query = st.text_input("Enter your question")

    if user_query:
        with st.spinner("Finding answer..."):
            relevant_chunks = get_relevant_chunks(vector_store, user_query)
            query_context = " ".join(relevant_chunks)
            query_summary = summarize_long_text(summarizer, query_context)
            st.subheader(" Answer")
            st.write(query_summary)
