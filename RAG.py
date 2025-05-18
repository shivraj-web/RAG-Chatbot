import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_hub import HuggingFaceHub
import numpy as np

# Hardcode your HuggingFace Hub token here for local run
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

st.set_page_config(page_title="America's Choice RAG Bot", page_icon="🤖")

st.title("📋 America's Choice RAG Chatbot")
st.markdown("""
Ask questions about America's Choice health plans. If your question is not answered based on the documentation, the bot will say "I don't know."
""")

@st.cache_resource
def load_vectorstore():
    # Load documents from PDFs and DOCX
    loaders = [
        PyMuPDFLoader("America's_Choice_2500_Gold_SOB (1) (1).pdf"),
        PyMuPDFLoader("America's_Choice_5000_Bronze_SOB (2).pdf"),
        PyMuPDFLoader("America's_Choice_5000_HSA_SOB (2).pdf"),
        PyMuPDFLoader("America's_Choice_7350_Copper_SOB (1) (1).pdf"),
        Docx2txtLoader("America's_Choice_Medical_Questions_-_Modified_(3) (1).docx")
    ]

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load a document: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Use HuggingFace sentence-transformer embeddings (free and cloud-hosted)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # Get text content from chunks
    texts = [doc.page_content for doc in chunks]

    # Embed texts as vectors
    embedding_vectors = embeddings.embed_documents(texts)

    # Convert embeddings to float32 numpy array
    embedding_vectors = np.array(embedding_vectors).astype("float32")

    # Create FAISS vectorstore from embeddings and documents
    db = FAISS.from_embeddings(embedding_vectors, chunks)
    return db

@st.cache_resource
def load_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Use HuggingFaceHub LLM (free hosted model) instead of local Ollama
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Example free model; change as needed
        model_kwargs={"temperature": 0, "max_length": 256},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return chain

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Searching documents..."):
        db = load_vectorstore()
        if db is not None:
            qa = load_qa_chain(db)
            try:
                result = qa.run(query)
                if result and ("coverage" in result.lower() or "plan" in result.lower() or "detego" in result.lower()):
                    st.success(result)
                else:
                    st.warning("I don't know.")
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.error("Document database not available. Please check file paths and formats.")
