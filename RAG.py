# File: app.py
import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# Set Streamlit page config
st.set_page_config(page_title="America's Choice RAG Bot", page_icon="🤖")

st.title("📋 America's Choice RAG Chatbot")
st.markdown("""
Ask questions about America's Choice health plans. If your question is not answered based on the documentation, the bot will say "I don't know."
""")

# Read Hugging Face token from Streamlit secrets
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

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

    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Use free Hugging Face sentence transformer model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store from the document chunks
    db = FAISS.from_documents(chunks, embeddings)
    return db

@st.cache_resource
def load_qa_chain(_db):
    retriever = _db.as_retriever(search_kwargs={"k": 4})

    # Replace local LLM with Hugging Face cloud model (free tier)
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Free model
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return chain

# Main user input interface
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
