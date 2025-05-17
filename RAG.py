# File: app.py
import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # ✅ Local model

st.set_page_config(page_title="America's Choice RAG Bot", page_icon="🤖")

st.title("📋 America's Choice RAG Chatbot")
st.markdown("""
Ask questions about America's Choice health plans. If your question is not answered based on the documentation, the bot will say "I don't know."
""")

@st.cache_resource
def load_vectorstore():
    # Load and process all documents
    loaders = []
    try:
        loaders = [
            PyMuPDFLoader("America's_Choice_2500_Gold_SOB (1) (1).pdf"),
            PyMuPDFLoader("America's_Choice_5000_Bronze_SOB (2).pdf"),
            PyMuPDFLoader("America's_Choice_5000_HSA_SOB (2).pdf"),
            PyMuPDFLoader("America's_Choice_7350_Copper_SOB (1) (1).pdf"),
            Docx2txtLoader("America's_Choice_Medical_Questions_-_Modified_(3) (1).docx")
        ]
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None

    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load a document: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # ✅ Replace OpenAI embeddings with local FAISS support (using raw text only for now)
    # You can optionally replace this with HuggingFaceEmbeddings or similar if needed
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(chunks, embeddings)
    return db

@st.cache_resource
def load_qa_chain(_db):  # <- Add underscore here
    retriever = _db.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="llama3.2"),
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
