import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

st.set_page_config(page_title="America's Choice RAG Bot", page_icon="🤖")

st.title("📋 America's Choice RAG Chatbot")
st.markdown("""
Ask questions about America's Choice health plans. If your question is not answered based on the documentation, the bot will say "I don't know."
""")

# Replace this token with your actual HuggingFace API token or load from .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN)

def query_hf_llm(prompt):
    try:
        response = client.text_generation(model="google/flan-t5-base", inputs=prompt, max_new_tokens=150)
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Error querying HuggingFace LLM: {e}")
        return ""

@st.cache_resource
def load_vectorstore():
    loaders = [
        PyMuPDFLoader("America's_Choice_2500_Gold_SOB (1) (1).pdf"),
        PyMuPDFLoader("America's_Choice_5000_Bronze_SOB (2).pdf"),
        PyMuPDFLoader("America's_Choice_5000_HSA_SOB (2).pdf"),
        PyMuPDFLoader("America's_Choice_7350_Copper_SOB (1) (1).pdf"),
        Docx2txtLoader("America's_Choice_Medical_Questions_-_Modified_(3) (1).docx")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db

@st.cache_resource
def load_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 4})

    # We create a wrapper for the HuggingFace Inference API call
    class HFLLMWrapper:
        def __call__(self, prompt):
            return query_hf_llm(prompt)

    llm = HFLLMWrapper()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

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
