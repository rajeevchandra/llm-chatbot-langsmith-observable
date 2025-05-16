# 🏋️ Gym Support Chatbot (Offline with Ollama + LangSmith)

This is a fully offline, AI-powered chatbot built using [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [LangSmith](https://smith.langchain.com/).  
It can answer gym-related queries based on your own uploaded `.txt` policy files.

---

### 💡 Features
- **RAG (Retrieval-Augmented Generation)** chatbot
- Runs 100% **offline** with `llama3` from **Ollama**
- **HuggingFace** embeddings (no cloud APIs required)
- **Streamlit UI** for easy interaction
- **LangSmith integration** for observability and feedback logging
- **Live document upload** support

---

### 🛠 Setup Instructions

#### 1. Install Requirements
```bash
pip install streamlit langchain faiss-cpu sentence-transformers
```

#### 2. Install & Run Ollama
Make sure [Ollama is installed](https://ollama.com/download) and running.

```bash
ollama run llama3
```

#### 3. Run the Streamlit App
```bash
cd app
streamlit run ui.py
```

---

### 📁 Project Structure
```
app/
├── data/
│   └── gym_policy.txt         # Sample gym policy
├── ui.py                      # Main Streamlit UI (Ollama + LangSmith)
├── main.py                    # CLI version (optional)
├── components/, utils/        # Reserved for extension
```

---

### 🧪 Sample Questions
- What are the gym hours on weekends?
- Can I bring a guest?
- Can I transfer my membership?

---

### 📈 LangSmith Integration
- Tracks all runs and latencies
- Records user feedback (`thumbs_up`, `thumbs_down`)
- Helps you trace & debug query paths

---

### ⚙️ .env-Free Deployment
No `.env` file or API key required for LLM or embeddings.

---

### 🧠 Credits
- 🧱 LangChain for chain composition
- 🧠 Ollama for local LLMs
- 🧬 HuggingFace for embeddings
- 📊 LangSmith for observability
- 🖼 Streamlit for UI

---

### 🤝 Contributing
Feel free to fork, enhance, and use this with your own private documents.

---

### 📬 Contact
For issues, open a GitHub issue or find me on the [LangChain Discord](https://discord.gg/langchain).
