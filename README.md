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



  ## 📂 Project Structure
engg-assist-bot/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies
│── .env.example # Sample config
│── README.md # This file
│── hyderabad_engineering_colleges.csv 
│── hyd_college_faq_extended.csv 



🚀 Setup

Clone repo

git clone https://github.com/YOURNAME/engg-assist-bot.git
cd engg-assist-bot

Create virtual env

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

Install dependencies

pip install -r requirements.txt

Configure .env

cp .env.example .env

Start Ollama

ollama serve

Run the app

streamlit run app.py
Open → http://localhost:8501

💡 Usage
Ask:

Best CSE colleges in Hyderabad with placements?

Fees for B.Tech near Kukatpally?

Which colleges accept TS EAMCET rank ~10k?

Upload more CSVs from the sidebar to expand knowledge.

⚠️ Notes
Answers are grounded only in provided CSVs.

Index persists in ./rag_index/.

Click Rebuild Index if you update CSVs.

Streamlit Cloud/HF Spaces won’t work unless Ollama is remote-exposed.

