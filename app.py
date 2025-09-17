import os
import textwrap
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# =========================
# Config (from Secrets/env)
# =========================
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
COLLECTION = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "doc_pages_e5s"))
TOP_K_DEFAULT = int(st.secrets.get("TOP_K", os.getenv("TOP_K", "5")))

LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", os.getenv("LLM_BASE_URL", ""))  # OpenAI-compatible
LLM_API_KEY  = st.secrets.get("LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
LLM_MODEL    = st.secrets.get("LLM_MODEL", os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"))

st.set_page_config(page_title="Corpus Agent", page_icon="🔎", layout="wide")
st.title("🔎 Indonesia Terrorism Researcher")
st.caption("Understanding terrorism in Indonesia, based on verified, selected sources. English & Bahasa Indonesia.")

# =========================
# Cached resources
# =========================
@st.cache_resource(show_spinner=True)
def get_embed_model():
    # Must match your indexing model
    m = SentenceTransformer("intfloat/multilingual-e5-small")
    m.max_seq_length = 512
    return m

@st.cache_resource(show_spinner=False)
def get_qdrant():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embed_model = get_embed_model()
client = get_qdrant()

# =========================
# --- Reranker (cross-encoder) ---
# =========================
@st.cache_resource(show_spinner=True)
def get_reranker():
    try:
        from FlagEmbedding import FlagReranker
        import torch
        use_fp16 = torch.cuda.is_available()  # fp16 if GPU; falls back on CPU
        return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=use_fp16)
    except Exception as e:
        st.warning(f"Reranker unavailable ({e}). Using vector search only.")
        return None

def rerank(query: str, points, keep: int = 12):
    """Reorder retrieved points by cross-encoder score, keep top N."""
    rr = get_reranker()
    if rr is None or not points:
        return points[:keep]
    pairs = [(f"query: {query}", (p.payload or {}).get("text", "")) for p in points]
    scores = rr.compute_score(pairs, normalize=True)  # higher = better
    ranked = [p for _, p in sorted(zip(scores, points), key=lambda x: x[0], reverse=True)]
    return ranked[:keep]

# =========================
# Retrieval utilities
# =========================
def search_points(query: str, top_k: int):
    qvec = embed_model.encode([query], normalize_embeddings=True, truncate=True)[0].tolist()
    res = client.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k, with_payload=True)
    return res

def build_context(points, max_chars=9000):
    """
    Turn top points into context with inline citation tags [S1], [S2]...
    Returns (context_text, citations_list)
    """
    contexts, citations = [], []
    total = 0
    for i, p in enumerate(points, 1):
        pay = p.payload or {}
        tag = f"S{i}"
        src = pay.get("source", "unknown")
        pages = f"{pay.get('page_start','?')}-{pay.get('page_end','?')}"
        txt = (pay.get("text","") or "").strip()
        snippet = txt[:1500]  # keep each snippet compact
        block = f"[{tag}] {snippet}"
        if total + len(block) > max_chars:
            break
        contexts.append(block)
        citations.append(f"[{tag}] {src} (pp. {pages})")
        total += len(block)
    return "\n\n".join(contexts), citations

def make_prompt(question: str, context: str, detail: str = "deep"):
    """
    detail: 'brief' | 'normal' | 'deep'
    """
    styles = {
        "brief":  "Write 3–5 sentences.",
        "normal": "Write 2–3 short paragraphs with citations.",
        "deep":   ("Write 4–6 short paragraphs with citations. Use this structure:\n"
                   "1) Overview\n2) Key developments/drivers\n3) Key actors & relationships\n"
                   "4) Timeline (brief chronological bullets)\n5) Gaps/uncertainties and caveats.")
    }
    system = (
        "You are a careful research assistant. Use ONLY the provided context. "
        "Cite sources inline using [S1], [S2], etc after claims. "
        "If the answer is not in the context, say you don’t know."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context (snippets with tags):\n{context}\n\n"
        f"Instructions:\n- {styles.get(detail, styles['deep'])}\n"
        f"- Keep claims tied to the snippets and include [S#] after the relevant sentences.\n"
        f"- Prefer concise, factual prose. Use bullets only where helpful."
    )
    return system, user

def call_llm(system: str, user: str, *, max_tokens: int = 1200, temperature: float = 0.2):
    if not (LLM_BASE_URL and LLM_API_KEY and LLM_MODEL):
        raise RuntimeError("LLM settings missing. Add LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in Secrets.")
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    if "openrouter.ai" in LLM_BASE_URL:
        headers["HTTP-Referer"] = "https://your-streamlit-app-url.example"
        headers["X-Title"] = "Corpus Agent"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    import requests
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# =========================
# UI
# =========================
tab1, tab2 = st.tabs(["Search", "Ask (Q&A)"])

with tab1:
    st.subheader("Semantic Search")
    with st.sidebar:
        st.markdown("**Collection:** `%s`" % COLLECTION)
    q = st.text_input("Search query", placeholder="e.g., perkembangan rekrutmen JAD 2016–2020 di Indonesia")
    top_k = st.slider("Top-K", 1, 20, TOP_K_DEFAULT, 1, key="search_k")
    if st.button("Search", type="primary") and q:
        with st.spinner("Searching..."):
            results = search_points(q, top_k)
        if not results:
            st.info("No results found.")
        else:
            st.success(f"Top {len(results)} results")
            for i, r in enumerate(results, 1):
                p = r.payload or {}
                source = p.get("source", "unknown")
                title  = p.get("title", "")
                pages  = f"{p.get('page_start','?')}-{p.get('page_end','?')}"
                text   = (p.get("text","") or "").strip()
                snippet = text[:800] + ("…" if len(text) > 800 else "")
                st.markdown(f"### {i}. {title or source}")
                st.markdown(f"**Source:** `{source}` **Pages:** `{pages}`")
                st.write(snippet)
                st.markdown("---")

with tab2:
    st.subheader("Ask the Agent (grounded Q&A)")
    q2 = st.text_input("Your question", placeholder="e.g., How did pro-ISIS recruitment evolve after 2016 in Indonesia?")

    colA, colB, colC = st.columns(3)
    with colA:
        top_k_qna = st.slider("Evidence passages (Top-K)", 5, 20, 12, 1)
    with colB:
        ctx_chars = st.slider("Context size (chars)", 3000, 20000, 9000, 1000)
    with colC:
        max_toks = st.slider("Max answer tokens", 256, 2400, 1200, 64)

    detail = st.selectbox("Answer depth", ["deep", "normal", "brief"], index=0)

    if st.button("Answer", type="primary") and q2:
        with st.spinner("Retrieving evidence..."):
            # fetch more than we need; we’ll keep the top_k_qna after context build if you add reranking later
            points = search_points(q2, max(20, top_k_qna))
        if not points:
            st.info("No evidence found in the index.")
        else:
            context, cites = build_context(points, max_chars=ctx_chars)
            try:
                sys_msg, user_msg = make_prompt(q2, context, detail=detail)
                with st.spinner("Writing grounded answer..."):
                    answer = call_llm(sys_msg, user_msg, max_tokens=max_toks, temperature=0.2)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("##### Sources")
                for c in cites[:top_k_qna]:
                    st.markdown("- " + c)
                with st.expander("Show retrieved snippets"):
                    for i, p in enumerate(points[:top_k_qna], 1):
                        pay = p.payload or {}
                        src = pay.get("source", "unknown")
                        pages = f"{pay.get('page_start','?')}-{pay.get('page_end','?')}"
                        text = (pay.get("text","") or "").strip()
                        st.markdown(f"**[S{i}]** `{src}` (pp. {pages})")
                        st.write(text[:1200] + ("…" if len(text) > 1200 else ""))
                        st.markdown("---")
            except Exception as e:
                st.error(f"LLM error: {e}")

# Diagnostics
with st.expander("Diagnostics"):
    try:
        cols = client.get_collections().collections
        st.write("Collections on cluster:", [c.name for c in cols])
        pts, _ = client.scroll(collection_name=COLLECTION, limit=1, with_payload=False)
        st.write(f"`{COLLECTION}` sample check:", "has data ✅" if pts else "0 points ❌")
        st.write("LLM base URL set:", bool(LLM_BASE_URL))
        st.write("LLM model:", LLM_MODEL)
    except Exception as e:
        st.error(f"Diagnostics error: {e}")
