from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import LangChainTracer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os

# Set environment variables (replace with your keys or use dotenv)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "GymSupportBot"

# Setup LangSmith Tracer
tracer = LangChainTracer(project_name="GymSupportBot")

# Load and split sample documents
loader = TextLoader("data/gym_policy.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Embeddings and vector store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding)

# Build RAG chain
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Run QA with LangSmith tracing
while True:
    query = input("Ask a gym-related question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    response = qa_chain.run({"query": query}, config={"callbacks": [tracer]})
    print("Answer:", response)
