from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyMuPDFLoader

def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text = " ".join(doc.page_content for doc in documents)
    return text

def split_text(text, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(chunks, embedding_model)

# Retrieve relevant chunks based on query
def get_relevant_chunks(vector_store, query, k=5):
    return [result.page_content for result in vector_store.similarity_search(query, k=k)]

# Summarization Pipeline
def summarize_long_text(text, chunk_size=800):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokens = text.split()
    summaries = []

    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i:i+chunk_size])
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    # Optionally, summarize the combined summaries again (map-reduce style)
    final_summary = summarizer(" ".join(summaries), max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return final_summary


# Main Logic
file_path = "iehp101.pdf"
document_text = load_pdf(file_path)
chunks = split_text(document_text)

# Create vector store for semantic search
vector_store = create_vector_store(chunks)

# Summarize entire document
full_text = " ".join(chunks[:5])  # Adjust as needed for shorter sections
document_summary = summarize_long_text(full_text)
print("Document Summary:", document_summary)

# Query-based retrieval and Q&A
query = "What Is Illness?"
relevant_chunks = get_relevant_chunks(vector_store, query)

# Summarize retrieved relevant chunks
query_context = " ".join(relevant_chunks)
query_summary = summarize_long_text(query_context)
print(f"Answer for '{query}':", query_summary)
