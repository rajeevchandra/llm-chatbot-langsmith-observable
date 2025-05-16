import streamlit as st
st.set_page_config(page_title="Gym Support Chatbot", layout="centered")

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks import LangChainTracer
from langchain_community.llms import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
import os
import tempfile

# Load environment
load_dotenv()
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "GymSupportBot"))

# Upload support
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
else:
    file_path = "data/gym_policy.txt"

# Safe load
docs = []
if os.path.exists(file_path):
    try:
        loader = TextLoader(file_path)
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("No default policy file found. Please upload a .txt file.")

# Chunking
split_docs = []
if docs:
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

# TF-IDF Embedding
class SklearnTfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def embed_documents(self, texts):
        return self.vectorizer.fit_transform(texts).toarray().tolist()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].tolist()

# Only proceed if docs loaded
if split_docs:
    texts = [doc.page_content for doc in split_docs]
    embedding = SklearnTfidfEmbeddings()
    vectorstore = FAISS.from_texts(texts, embedding)
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.title("🏋️ Gym Support Chatbot (Custom TF-IDF)")
    st.markdown("Ask anything about your uploaded or default gym policy.")

    query = st.text_input("Your question")
    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke({"query": query}, config={"callbacks": [tracer]})
        st.success(answer)

        # Feedback (saved locally)
        feedback = st.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True)
        if feedback:
            score = "👍 Yes" if feedback == "👍 Yes" else "👎 No"
            with open("feedback_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{query} => {score}\n")
else:
    st.info("📂 Please upload a document in the sidebar to begin.")
