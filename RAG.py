import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# UI Setup good
st.set_page_config(page_title="America's Choice RAG Bot", page_icon="🤖")
st.title("📋 America's Choice RAG Chatbot")
st.markdown("""
Ask questions about America's Choice health plans. If your question is not answered based on the documentation, the bot will say "I don't know."
""")

# Load vectorstore
@st.cache_resource(show_spinner="Loading documents...")
def load_vectorstore():
    try:
        loaders = [
            PyMuPDFLoader("America's_Choice_2500_Gold_SOB.pdf"),
            PyMuPDFLoader("America's_Choice_5000_Bronze_SOB.pdf"),
            PyMuPDFLoader("America's_Choice_5000_HSA_SOB.pdf"),
            PyMuPDFLoader("America's_Choice_7350_Copper_SOB.pdf"),
            Docx2txtLoader("America's_Choice_Medical_Questions_Modified.docx")
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        db = FAISS.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

# Load QA Chain
@st.cache_resource(show_spinner="Initializing QA system...")
def load_qa_chain(_db):
    try:
        retriever = _db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )

        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(
            "text2text-generation",
            model=model_id,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            do_sample=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        return qa
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Load once
if "db" not in st.session_state:
    st.session_state.db = load_vectorstore()

if "qa_chain" not in st.session_state and st.session_state.db:
    st.session_state.qa_chain = load_qa_chain(st.session_state.db)

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and answer
prompt = st.text_input("Ask about America's Choice health plans:")
if prompt :
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                if not st.session_state.get("qa_chain"):
                    st.error("QA system not ready")
                    st.stop()

                response = st.session_state.qa_chain({"query": prompt})
                answer = response.get('result', '') if isinstance(response, dict) else str(response)
                answer = answer.replace('\n', ' ').strip()

                keywords = ["deductible", "coverage", "plan", "$", "dollar", "detego", "5000", "10000"]
                if any(kw in answer.lower() for kw in keywords):
                    st.markdown(answer)
                else:
                    st.warning("I don't know.")

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
