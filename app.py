"""Streamlit Web Application for NSG Academy Knowledge Assistant.

Provides an intuitive local-first interface to:
1. Index PDF documents into ChromaDB.
2. Query using Grounded RAG, Baseline LLM-Only, or Side-by-Side comparison.
3. Inspect citations, page references, and similarity retrieval diagnostics.
"""

from pathlib import Path
import time
import streamlit as st

from src.config import load_config
from src.rag_pipeline import RAGPipeline, PipelineAnswer


# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NSG Academy Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, professional academic aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #F1F5F9;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
        margin-right: 8px;
    }
    .citation-tag {
        display: inline-block;
        padding: 3px 8px;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 4px;
        color: #1D4ED8;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    """Initializes and caches the RAGPipeline instance across app reruns."""
    return RAGPipeline()


pipeline = get_pipeline()
config = pipeline.config


# -----------------------------------------------------------------------------
# Sidebar: Document Management & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.write(f"**Model:** `{config.gemini_model_name}`")
    st.write(f"**API Key:** `{config.masked_api_key()}`")
    st.write(f"**Embeddings:** `{config.embedding_model_name}`")
    
    current_count = pipeline.vector_store.count()
    if current_count > 0:
        st.success(f"📦 Index Active: **{current_count}** chunks stored")
    else:
        st.warning("⚠️ Index Empty: Please index a document below.")

    st.markdown("---")
    st.header("📄 Document Ingestion")

    # Document folder discovery
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    existing_pdfs = list(docs_dir.glob("*.pdf"))
    pdf_names = [p.name for p in existing_pdfs]

    selected_existing = st.selectbox(
        "Choose existing local PDF:",
        options=["(Select a file)"] + pdf_names,
        index=1 if pdf_names else 0,
    )

    uploaded_file = st.file_uploader(
        "Or upload a new PDF:",
        type=["pdf"],
        help="Upload a PDF to store in data/documents and index into ChromaDB.",
    )

    # Handle document upload
    target_pdf_path: Optional[Path] = None
    if uploaded_file is not None:
        save_path = docs_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_pdf_path = save_path
        st.info(f"Uploaded `{uploaded_file.name}` to `data/documents`")
    elif selected_existing != "(Select a file)":
        target_pdf_path = docs_dir / selected_existing

    # Index action
    if target_pdf_path and target_pdf_path.is_file():
        if st.button("⚡ Index Selected Document", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting, chunking, and embedding '{target_pdf_path.name}'..."):
                try:
                    result = pipeline.index_document(str(target_pdf_path))
                    st.success(
                        f"Indexed {result['chunks_indexed']} chunks from "
                        f"{result['pages_loaded']} pages in {result['indexing_latency_seconds']}s!"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    st.markdown("---")
    # Reset index option
    with st.expander("🗑️ Database Maintenance"):
        st.write("Wipes the persistent ChromaDB index to rebuild fresh.")
        if st.button("Reset Vector Index", type="secondary", use_container_width=True):
            pipeline.vector_store.reset_index()
            st.warning("Vector index cleared!")
            st.rerun()

    # Query Configuration
    st.markdown("---")
    st.header("🔍 Retrieval Settings")
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=5, value=config.default_top_k)


# -----------------------------------------------------------------------------
# Main Area: Question Answering & Comparison
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📚 NSG Academy Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Local-first PDF Question Answering system with verifiable provenance, page citations, '
    'and Baseline vs. Grounded RAG comparative evaluation.'
    '</div>',
    unsafe_allow_html=True,
)

# Mode Selection
mode_choice = st.radio(
    "Select Operating Mode:",
    options=[
        "🟢 RAG (Context-Grounded)",
        "🔵 LLM-Only (Baseline)",
        "⚖️ Side-by-Side Comparison (RAG vs Baseline)",
    ],
    horizontal=True,
    help="Compare direct pre-trained LLM answers against document-grounded RAG answers.",
)

# Question Input
question_input = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the primary goal of Assignment 01?",
    key="user_query",
)

col_ask, col_clear = st.columns([1, 8])
with col_ask:
    ask_button = st.button("Ask Assistant", type="primary")


# -----------------------------------------------------------------------------
# Execution & Results Rendering
# -----------------------------------------------------------------------------
if ask_button:
    if not question_input.strip():
        st.warning("Please type a question before submitting.")
    elif not config.is_api_key_configured:
        st.error("Gemini API key is not configured. Please add `GEMINI_API_KEY` to your local `.env` file.")
    else:
        # ---------------------------------------------------------------------
        # Side-by-Side Comparison Mode
        # ---------------------------------------------------------------------
        if "Side-by-Side" in mode_choice:
            st.subheader("⚖️ Side-by-Side Evaluation")
            col_rag, col_base = st.columns(2)

            with col_rag:
                st.markdown("### 🟢 Grounded RAG")
                with st.spinner("Retrieving chunks and generating grounded answer..."):
                    rag_ans = pipeline.answer_question(question_input, mode="RAG", top_k=top_k)

                st.markdown(rag_ans.answer)
                st.markdown(
                    f"<span class='metric-badge'>⏱️ Total: {rag_ans.total_latency_seconds:.2f}s</span>"
                    f"<span class='metric-badge'>🔍 Retrieval: {rag_ans.retrieval_latency_seconds:.2f}s</span>"
                    f"<span class='metric-badge'>🤖 Gen: {rag_ans.generation_latency_seconds:.2f}s</span>",
                    unsafe_allow_html=True,
                )
                if rag_ans.citations:
                    st.markdown("**Citations:**")
                    for c in rag_ans.citations:
                        st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)

            with col_base:
                st.markdown("### 🔵 Baseline (LLM-Only)")
                with st.spinner("Generating baseline answer without context..."):
                    base_ans = pipeline.answer_question(question_input, mode="LLM-Only")

                st.markdown(base_ans.answer)
                st.markdown(
                    f"<span class='metric-badge'>⏱️ Total: {base_ans.total_latency_seconds:.2f}s</span>"
                    f"<span class='metric-badge'>🤖 Gen: {base_ans.generation_latency_seconds:.2f}s</span>",
                    unsafe_allow_html=True,
                )
                st.info("ℹ️ Baseline has no access to document chunks or page citations.")

            # Diagnostic Expander for RAG
            if rag_ans.retrieved_chunks:
                with st.expander("🔍 View Retrieved Context & Similarity Diagnostics"):
                    for i, chunk in enumerate(rag_ans.retrieved_chunks, 1):
                        st.markdown(
                            f"<div class='chunk-box'>"
                            f"<b>Chunk #{i}</b> | Page: <b>{chunk.primary_page}</b> | "
                            f"Similarity: <b>{chunk.similarity:.4f}</b> (Dist: {chunk.distance:.4f})<br>"
                            f"<i>ID: {chunk.chunk_id}</i><br><br>"
                            f"{chunk.text}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # ---------------------------------------------------------------------
        # Single Mode (RAG or Baseline)
        # ---------------------------------------------------------------------
        else:
            mode_tag = "RAG" if "RAG" in mode_choice else "LLM-Only"
            with st.spinner(f"Generating answer in {mode_tag} mode..."):
                answer_obj = pipeline.answer_question(question_input, mode=mode_tag, top_k=top_k)

            st.markdown("### 💬 Answer")
            st.markdown(answer_obj.answer)

            # Metadata and Latency metrics
            st.markdown(
                f"<span class='metric-badge'>⏱️ Total: {answer_obj.total_latency_seconds:.2f}s</span>"
                f"<span class='metric-badge'>🔍 Retrieval: {answer_obj.retrieval_latency_seconds:.2f}s</span>"
                f"<span class='metric-badge'>🤖 LLM Gen: {answer_obj.generation_latency_seconds:.2f}s</span>",
                unsafe_allow_html=True,
            )

            # Citations for RAG
            if answer_obj.citations:
                st.markdown("**Source Citations:**")
                for c in answer_obj.citations:
                    st.markdown(f"<span class='citation-tag'>📄 {c}</span>", unsafe_allow_html=True)

            # Expandable Debug View
            if answer_obj.retrieved_chunks:
                with st.expander("🔍 Retrieved Document Excerpts & Provenance"):
                    for i, chunk in enumerate(answer_obj.retrieved_chunks, 1):
                        st.markdown(
                            f"<div class='chunk-box'>"
                            f"<b>Excerpt #{i}</b> | Document: <code>{chunk.doc_name}</code> | "
                            f"Page: <b>{chunk.primary_page}</b> | Similarity: <b>{chunk.similarity:.4f}</b><br>"
                            f"<i>Chunk ID: {chunk.chunk_id}</i><br><br>"
                            f"{chunk.text}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
