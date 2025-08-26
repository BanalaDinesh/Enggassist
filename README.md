# 🎓 Engg Assist – Hyderabad B.Tech Chatbot 🤖

An AI-powered admissions assistant for students exploring **engineering colleges in Hyderabad**.  
This chatbot uses **RAG (Retrieval-Augmented Generation)** with FAISS + Ollama models to answer queries about colleges, fees, placements, exams, and more.

---

## ✨ Features
- 💬 Conversational chatbot (Streamlit UI)
- 🔎 Vector search with **FAISS** over CSV data
- 📊 Grounded answers from your uploaded college datasets
- 📁 Upload new CSVs dynamically
- 🧠 Uses **Ollama** models:
  - Generator: `llama3.1:8b` (customizable)
  - Embeddings: `nomic-embed-text` (customizable)

---

## ⚙️ Requirements
- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- Models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
