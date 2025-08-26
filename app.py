import os
import json
import time
import uuid
import shutil
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# === .env support ===
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------
# CONFIG
# ------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  
INDEX_DIR = os.getenv("INDEX_DIR", "rag_index")
METADATA_JSON = os.path.join(INDEX_DIR, "meta.json")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
DEFAULT_CSVS = [
    "hyderabad_engineering_colleges.csv",
    "hyd_college_faq_extended.csv",
]
TOP_K_DEFAULT = 6

# ------------------------------
# OLLAMA CLIENT 
# ------------------------------
import requests

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def embeddings(self, model: str, input_text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        resp = requests.post(url, json={"model": model, "input": input_text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding")

    def chat(self, model: str, messages: List[Dict], temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

ollama = OllamaClient()

# ------------------------------
# VECTOR STORE (FAISS)
# ------------------------------
try:
    import faiss  # type: ignore
except Exception as e:
    st.stop()


def ensure_index_dir():
    os.makedirs(INDEX_DIR, exist_ok=True)


def normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    return " ".join(x.split())


def df_to_corpus(df: pd.DataFrame) -> List[Dict]:
    """Convert any CSV into a per-row document with concatenated fields.
    Returns list of {id, text, metadata}.
    """
    docs = []
    for i, row in df.iterrows():
        meta = {col: normalize_str(row[col]) for col in df.columns}
        # Build a readable blob – prioritize common columns but include all
        parts = []
        for key in [
            "college_name", "name", "institute", "university", "affiliation", "location",
            "city", "area", "address", "pincode", "district", "state",
            "courses", "branches", "departments", "programs", "specializations",
            "fees", "tution_fee", "tuition_fee", "hostel_fee",
            "ranking", "nirf_rank", "naac", "nba",
            "admission", "eligibility", "entrance_exam",
            "website", "email", "phone", "contact",
            "scholarships", "placements", "companies", "average_package",
        ]:
            if key in meta and meta[key]:
                parts.append(f"{key.replace('_',' ').title()}: {meta[key]}")
        # Fallback: add any missing fields generically
        for col in df.columns:
            if col not in meta:
                continue
            val = meta[col]
            if val and all(val not in p for p in parts):
                parts.append(f"{col.replace('_',' ').title()}: {val}")

        text = "\n".join(parts) if parts else "\n".join([f"{k}: {v}" for k, v in meta.items() if v])
        docs.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": meta,
        })
    return docs


def embed_corpus(docs: List[Dict]) -> np.ndarray:
    vectors = []
    for d in docs:
        vec = ollama.embeddings(EMBED_MODEL, d["text"])  
        vectors.append(vec)
    return np.array(vectors, dtype="float32")


def save_index(index, docs: List[Dict]):
    ensure_index_dir()
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def build_or_load_index(csv_paths: List[str], progress_notice=True) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_JSON):
        try:
            index, docs = load_index()
            if progress_notice:
                st.toast("Index loaded from disk", icon="✅")
            return index, docs
        except Exception:
            # Corrupt index – rebuild
            shutil.rmtree(INDEX_DIR, ignore_errors=True)

    # Build from scratch
    ensure_index_dir()
    all_df = []
    for p in csv_paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            if len(df) > 0:
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                all_df.append(df)
    if not all_df:
        st.warning("No CSVs found. Upload in the sidebar, or place your CSVs next to app.py")
        return None, []

    merged = pd.concat(all_df, axis=0, ignore_index=True)
    docs = df_to_corpus(merged)

    # Build vectors (inner product; normalize to emulate cosine)
    with st.spinner("Embedding rows with Ollama (first time only)…"):
        X = embed_corpus(docs)
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        save_index(index, docs)
    st.toast("Index built & saved", icon="🎉")
    return index, docs


def search(index, docs: List[Dict], query: str, k: int = TOP_K_DEFAULT) -> List[Dict]:
    q_vec = np.array([ollama.embeddings(EMBED_MODEL, query)], dtype="float32")
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(q_vec, k)
    results = []
    for rank, idx in enumerate(I[0].tolist()):
        if idx < 0 or idx >= len(docs):
            continue
        hit = docs[idx]
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "id": hit["id"],
            "text": hit["text"],
            "metadata": hit.get("metadata", {}),
        })
    return results


SYSTEM_PROMPT = (
    "You are a helpful admissions advisor for students exploring engineering colleges in Hyderabad. "
    "Answer clearly and concisely. When you reference facts, ground them strictly in the provided CONTEXT. "
    "Prefer concrete details (fees, branches, location, admission tests, accreditation, placements) if present. "
    "If the answer is not in the context, say you don't have that info and suggest how the student can proceed. "
    "Provide a short, bulleted summary and include a 'How to Decide' checklist when appropriate. "
)


def build_context_block(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[DOC {i}]\n{h['text']}")
    return "\n\n".join(blocks)


def generate_answer(user_query: str, hits: List[Dict], chat_history: List[Dict], temperature: float = 0.2) -> str:
    context_block = build_context_block(hits)
    content = (
        f"CONTEXT (top {len(hits)} docs):\n{context_block}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        "Instructions: Use only the context for factual claims about colleges. If unsure, say so."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *chat_history, 
        {"role": "user", "content": content},
    ]
    answer = ollama.chat(GENERATION_MODEL, messages, temperature=temperature, top_p=0.9, max_tokens=700)
    return answer


# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="🎓 Engg Assist - Hyderabad B.Tech Chatbot 🤖", page_icon="🎓", layout="wide")

st.title("🎓 Engg Assist - Hyderabad B.Tech Chatbot 🤖")

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Ollama must be running locally. Models: llama3.1:8b & nomic-embed-text")
    gen_model = st.text_input("Generator model", value=GENERATION_MODEL)
    emb_model = st.text_input("Embedding model", value=EMBED_MODEL)
    if gen_model != GENERATION_MODEL:
        GENERATION_MODEL = gen_model
    if emb_model != EMBED_MODEL:
        EMBED_MODEL = emb_model

    top_k = st.slider("Retriever: top_k", min_value=3, max_value=15, value=TOP_K_DEFAULT, step=1)
    temp = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()
    st.subheader("📁 Data Sources")
    st.write("Place your CSVs next to `app.py` or upload below.")

    uploaded = st.file_uploader("Upload additional CSVs", type=["csv"], accept_multiple_files=True)
    extra_paths = []
    if uploaded:
        os.makedirs("uploads", exist_ok=True)
        for up in uploaded:
            p = os.path.join("uploads", up.name)
            with open(p, "wb") as f:
                f.write(up.read())
            extra_paths.append(p)
        st.success(f"Saved {len(uploaded)} file(s) to ./uploads")

    st.caption("Index will auto-build the first time or when you click Rebuild.")
    rebuild = st.button("🔁 Rebuild Index")

# Persist chat in session
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content}

if rebuild:
    shutil.rmtree(INDEX_DIR, ignore_errors=True)

# Build / load index
all_paths = [p for p in DEFAULT_CSVS if os.path.exists(p)] + extra_paths
index_docs_tuple = build_or_load_index(all_paths)
if index_docs_tuple == (None, []):
    st.stop()
index, docs = index_docs_tuple

# Chat input
query = st.chat_input("Ask about colleges, branches, fees, exams, location, placements…")

# Optional starter tips
with st.expander("💡 Sample questions", expanded=False):
    st.markdown(
        """
        - Best CSE colleges in Hyderabad with strong placements?  
        - Compare JNTU-affiliated vs autonomous colleges for ECE.  
        - Fees and eligibility for B.Tech in top private colleges near Kukatpally.  
        - Which colleges accept TS EAMCET rank around 10,000?  
        - NAAC/NBA accredited colleges with hostel facilities.  
        """
    )

# Display previous messages
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if query:
    st.session_state.chat.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            hits = search(index, docs, query, k=top_k)
            answer = generate_answer(query, hits, chat_history=st.session_state.chat[:-1], temperature=temp)
            st.markdown(answer)

            # Show sources panel
            with st.expander("🔎 Sources (top matches)", expanded=False):
                for h in hits:
                    md = h["metadata"]
                    title = md.get("college_name") or md.get("name") or md.get("institute") or "Document"
                    subtitle = md.get("location") or md.get("city") or "Hyderabad"
                    st.markdown(f"**{h['rank']}. {title}** — {subtitle}")
                    st.caption(f"score: {h['score']:.3f}")
                    # Print key fields if available
                    cols_to_show = [
                        "address", "pincode", "courses", "departments", "branches",
                        "admission", "entrance_exam", "eligibility", "fees", "average_package",
                        "website", "phone", "email",
                    ]
                    kvs = []
                    for k in cols_to_show:
                        v = md.get(k)
                        if v:
                            kvs.append(f"- **{k.title().replace('_',' ')}:** {v}")
                    if kvs:
                        st.markdown("\n".join(kvs))
                    st.markdown("---")

            st.session_state.chat.append({"role": "assistant", "content": answer})

st.caption("Made with Ollama + FAISS + Streamlit. First run may take time to embed.")