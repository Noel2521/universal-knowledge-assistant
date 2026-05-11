import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

# ── Page Setup ────────────────────────────────────────
st.set_page_config(
    page_title="Universal Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Universal Knowledge Assistant")
st.caption("Powered by NovaTech AI | LangChain + ChromaDB + Ollama")

# ── Sidebar: Document Upload ──────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or CSV",
        type=["pdf", "docx", "csv"]
    )

    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Processing document..."):
                response = requests.post(
                    f"{API_BASE_URL}/ingest",
                    files={"file": uploaded_file}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(data["message"])
                    st.info(f"📄 Pages loaded: {data['documents_loaded']}")
                    st.info(f"🔪 Chunks created: {data['chunks_created']}")
                else:
                    st.error("Failed to ingest document")

# ── Main Area: Tabs ───────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask a Question", "🗣️ Conversation"])

# ── Tab 1: Single Question ────────────────────────────
with tab1:
    st.subheader("Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is NovaTech's P99 latency target?"
    )

    if st.button("Get Answer", key="query_btn"):
        if not question:
            st.warning("Please enter a question first!")
        else:
            with st.spinner("Searching documents..."):
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"question": question}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### 📝 Answer")
                    st.write(data["answer"])
                    st.markdown(f"**Confidence Score: {data['confidence']}**")
        
                    # ── Sources ───────────────────────
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(data["sources"]):
                        with st.expander(f"Source {i+1}: {source['document_name']}"):
                            if source["page"]:
                                st.write(f"**Page:** {source['page']}")
                            st.write(f"**Relevance:** {source['relevance_score']}")
                            st.write(f"**Preview:** {source['chunk_preview']}")
                else:
                    st.error("Failed to get answer. Are documents ingested?")


# ── Tab 2: Conversation ───────────────────────────────
with tab2:
    st.subheader("Multi-turn Conversation")
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = "session_001"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a follow-up question...")
    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/conversation",
                json={
                    "question": user_input,
                    "conversation_id": st.session_state.conversation_id
                }
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": data["answer"]}
                )
                st.rerun()
            else:
                st.error("Failed to get response.")