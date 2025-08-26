# 🎓 Engg Assist – Hyderabad B.Tech Chatbot 🤖


Engg Assist is an AI-powered chatbot that helps students explore **Hyderabad engineering colleges**, including details such as:
- College information
- Branches offered
- Fees
- Cutoffs
- Placements
- Accreditations
- Affiliation & Rankings

It uses **LangChain + Ollama (Llama3 model) + FAISS** to create a **RAG (Retrieval-Augmented Generation)** system with a **Gradio web UI**.

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

---  

## 📂 Project Structure

engg-assist-bot/  
│── app.py # Main Streamlit app  
│── requirements.txt # Dependencies  
│── .env.example # Sample config  
│── README.md # This file  
│── hyderabad_engineering_colleges.csv  
│── hyd_college_faq_extended.csv  
│── rag_index/ # Auto-created FAISS index  


---

## 🚀 Setup

1. Clone repo
```bash
git clone https://github.com/BanalaDinesh/Enggassist
cd engg-assist-bot 
```
2. Create virtual env
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure .env  
Copy `.env.example` to `.env` and update values as needed:
```bash
cp .env.example .env
```

5. Start Ollama
```bash
ollama serve
```

6. Run the app
```bash 
streamlit run app.py
```

---


## 💡 Usage Ask:

-Best CSE colleges in Hyderabad with placements?  
-Fees for B.Tech near Kukatpally?  
-Which colleges are nearby Ghatkesar?  
-Upload more CSVs from the sidebar to expand knowledge.  

---

## ⚠️ Notes
-Answers are grounded only in provided CSVs.  
-Index persists in ./rag_index/.  
-Click Rebuild Index if you update CSVs.  
-Streamlit Cloud/HF Spaces won’t work unless Ollama is remote-exposed.  



