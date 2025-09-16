import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
COLLECTION = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "doc_chunks"))
TOP_K_DEFAULT = int(st.secrets.get("TOP_K", os.getenv("TOP_K", "5")))

st.set_page_config(page_title="Corpus Search", page_icon="🔎", layout="wide")
st.title("🔎 Corpus Search (Qdrant + bge-m3)")
st.caption("Type a question or keywords. Works in English and Bahasa Indonesia.")

@st.cache_resource(show_spinner=True)
def get_model():
    model = SentenceTransformer("BAAI/bge-m3")   # must match your indexing model
    model.max_seq_length = 8192
    return model

@st.cache_resource(show_spinner=False)
def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

with st.sidebar:
    st.subheader("Settings")
    top_k = st.slider("Results to show (Top-K)", 1, 20, TOP_K_DEFAULT, 1)
    st.markdown(f"**Collection:** `{COLLECTION}`")

query = st.text_input("Your query", placeholder="e.g., perkembangan rekrutmen JAD 2016–2020 di Indonesia")
go = st.button("Search", type="primary") or bool(query)

if not QDRANT_URL or not QDRANT_API_KEY:
    st.warning("Add QDRANT_URL and QDRANT_API_KEY in Streamlit Secrets after deploy.")

if go and query:
    with st.spinner("Searching..."):
        try:
            model = get_model()
            client = get_client()
            qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
            res = client.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k, with_payload=True)
        except Exception as e:
            st.error(f"Search error: {e}")
            res = []

    if not res:
        st.info("No results found.")
    else:
        st.success(f"Top {len(res)} results")
        for i, r in enumerate(res, 1):
            p = r.payload or {}
            source = p.get("source", "unknown")
            title  = p.get("title", "")
            pages
