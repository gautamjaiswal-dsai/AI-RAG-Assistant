import streamlit as st
import os
import re
import tempfile
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

# =========================================
# LOAD ENV
# =========================================
load_dotenv()

# =========================================
# CONFIG
# =========================================
DB_DIR = "chroma_db"

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

.stApp { background-color: #0f1117; }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2a2d3e;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}

/* Stat cards */
.stat-row { display:flex; gap:10px; margin-bottom:12px; }
.stat-card {
    flex:1; background:#262730; border:1px solid #3a3d4e;
    border-radius:10px; padding:12px 10px; text-align:center;
}
.stat-card .num { font-size:26px; font-weight:700; color:#7c8cf8; line-height:1.1; }
.stat-card .lbl { font-size:11px; color:#8892a4; margin-top:3px; letter-spacing:0.04em; }

/* Chat title */
.chat-title {
    display:flex; align-items:center; gap:10px;
    padding-bottom:14px; border-bottom:1px solid #2a2d3e; margin-bottom:20px;
}
.chat-title h2 { font-size:18px; font-weight:600; color:#e2e8f0; margin:0; }
.chat-title span { font-size:12px; color:#5a6478; }

/* Empty state */
.empty-state { text-align:center; padding:60px 20px; }
.empty-state .icon { font-size:48px; margin-bottom:12px; }
.empty-state p { font-size:14px; color:#5a6478; }

/* ── USER bubble ── */
.msg-user {
    display:flex; justify-content:flex-end;
    align-items:flex-end; gap:10px; margin-bottom:16px;
}
.msg-user .av {
    width:32px; height:32px; border-radius:50%;
    background:#3b4fd8; display:flex; align-items:center;
    justify-content:center; font-size:12px; font-weight:700;
    color:#fff; flex-shrink:0;
}
.msg-user .bubble {
    background:#3b4fd8; color:#e8ecff;
    padding:11px 16px; border-radius:18px 18px 4px 18px;
    max-width:72%; font-size:14px; line-height:1.65;
    word-break:break-word;
}

/* ── AI bubble ── */
.msg-ai {
    display:flex; align-items:flex-start;
    gap:10px; margin-bottom:16px;
}
.msg-ai .av {
    width:32px; height:32px; border-radius:50%;
    background:#262730; border:1px solid #3a3d4e;
    display:flex; align-items:center; justify-content:center;
    font-size:16px; flex-shrink:0; margin-top:2px;
}
.msg-ai .bubble {
    background:#1e2030; border:1px solid #2e3147;
    color:#d4d8e8; padding:13px 17px;
    border-radius:4px 18px 18px 18px;
    max-width:82%; font-size:14px; line-height:1.75;
    word-break:break-word;
}

/* Markdown inside AI bubble */
.msg-ai .bubble p  { margin:0 0 8px; }
.msg-ai .bubble p:last-of-type { margin-bottom:0; }
.msg-ai .bubble ul, .msg-ai .bubble ol {
    margin:6px 0 10px 18px; padding:0;
}
.msg-ai .bubble li { margin-bottom:4px; }
.msg-ai .bubble strong { color:#a5b4fc; font-weight:600; }
.msg-ai .bubble code {
    background:#262730; color:#7dd3fc;
    padding:2px 6px; border-radius:4px; font-size:13px;
}
.msg-ai .bubble pre {
    background:#262730; border:1px solid #3a3d4e;
    border-radius:8px; padding:12px; overflow-x:auto;
    margin:10px 0;
}
.msg-ai .bubble pre code { background:none; padding:0; }
.msg-ai .bubble h1,.msg-ai .bubble h2,.msg-ai .bubble h3 {
    color:#c7d2fe; margin:12px 0 6px;
}

/* Source chips */
.sources { margin-top:12px; display:flex; flex-wrap:wrap; gap:6px; }
.src-chip {
    background:#12131f; border:1px solid #2e3a5e;
    border-radius:20px; padding:3px 10px;
    font-size:11px; color:#7c8cf8;
    display:inline-flex; align-items:center; gap:4px;
}

/* Buttons */
.stButton > button {
    border-radius:8px !important;
    font-size:13px !important;
    font-weight:500 !important;
}

/* Divider */
hr { border-color:#2a2d3e !important; margin:12px 0 !important; }

/* Chat input — full override */
[data-testid="stChatInput"] > div {
    background:#3a3f5c !important;
    border:1.5px solid #5a5f9f !important;
    border-radius:14px !important;
    box-shadow:none !important;
    outline:none !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color:#7c8cf8 !important;
    box-shadow:0 0 0 3px rgba(124,140,248,0.2) !important;
}
[data-testid="stChatInput"] textarea {
    background:#3a3f5c !important;
    color:#f4f6ff !important;
    font-size:14px !important;
    caret-color:#a5b4fc !important;
    border:none !important;
    outline:none !important;
    box-shadow:none !important;
    -webkit-text-fill-color:#f4f6ff !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color:#9099c0 !important;
    -webkit-text-fill-color:#9099c0 !important;
    opacity:1 !important;
}
[data-testid="stChatInput"] button {
    background:#3b4fd8 !important;
    border-radius:10px !important;
    border:none !important;
    color:#fff !important;
}

.stSpinner > div { color:#7c8cf8 !important; }
.stAlert { border-radius:10px !important; font-size:13px !important; }

</style>
""", unsafe_allow_html=True)


# =========================================
# MARKDOWN → HTML CONVERTER
# Converts LLM markdown output to safe HTML
# =========================================
def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for display inside unsafe_allow_html blocks."""
    # Escape HTML special chars first (except we'll add tags back)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Code blocks (``` ... ```) — must come before inline code
    text = re.sub(
        r'```(?:\w+)?\n(.*?)```',
        lambda m: f'<pre><code>{m.group(1)}</code></pre>',
        text, flags=re.DOTALL
    )

    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

    # Italic *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Numbered lists  (1. item)
    def replace_ol(m):
        items = re.findall(r'^\d+\.\s+(.+)$', m.group(0), re.MULTILINE)
        lis = "".join(f"<li>{i}</li>" for i in items)
        return f"<ol>{lis}</ol>"
    text = re.sub(r'((?:^\d+\..+\n?)+)', replace_ol, text, flags=re.MULTILINE)

    # Unordered lists  (- item  or  * item)
    def replace_ul(m):
        items = re.findall(r'^[-*]\s+(.+)$', m.group(0), re.MULTILINE)
        lis = "".join(f"<li>{i}</li>" for i in items)
        return f"<ul>{lis}</ul>"
    text = re.sub(r'((?:^[-*]\s.+\n?)+)', replace_ul, text, flags=re.MULTILINE)

    # Paragraphs — wrap non-tag lines
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('<h', '<ul', '<ol', '<pre', '<li')):
            result.append(stripped)
        else:
            result.append(f'<p>{stripped}</p>')
    return '\n'.join(result)


# =========================================
# SESSION STATE
# =========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None

if "uploaded_pdf_names" not in st.session_state:
    st.session_state.uploaded_pdf_names = []


# =========================================
# EMBEDDINGS
# =========================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = get_embeddings()


# =========================================
# LOAD DB
# =========================================
@st.cache_resource
def load_db():
    if not os.path.exists(DB_DIR):
        return None
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

vectorstore = load_db()


# =========================================
# LLM
# =========================================
llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct",
    temperature=0.2
)


# =========================================
# PROMPT
# =========================================
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant for a PDF Q&A system.

RULES:
- Use ONLY the provided context to answer.
- Do NOT use outside knowledge.
- If the answer is not found in context, say: "I could not find the answer in the document."
- Format your answers clearly using markdown (bold for key terms, numbered/bullet lists where helpful).
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}"""
    )
])


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:

    st.markdown("## 📚 AI RAG Assistant")
    st.markdown("---")

    # Stats — use stored PDF names for accurate count
    pdf_count = len(st.session_state.uploaded_pdf_names)
    chat_count = len(st.session_state.chat_history)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="num">{pdf_count}</div>
            <div class="lbl">📄 PDFs</div>
        </div>
        <div class="stat-card">
            <div class="num">{chat_count}</div>
            <div class="lbl">💬 Chats</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload PDF ──
    st.markdown("### 📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")

        if st.button("➕ Add to Knowledge Base", use_container_width=True):
            with st.spinner("Processing PDF..."):

                # Save temp file — store real filename in metadata
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                    prefix=uploaded_file.name.replace(".pdf", "") + "_"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = tmp.name

                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                # Patch source metadata to real filename
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100
                )
                chunks = splitter.split_documents(docs)

                if vectorstore is None:
                    new_vs = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        persist_directory=DB_DIR
                    )
                    new_vs.persist()
                else:
                    vectorstore.add_documents(chunks)
                    vectorstore.persist()

                # Track PDF name
                if uploaded_file.name not in st.session_state.uploaded_pdf_names:
                    st.session_state.uploaded_pdf_names.append(uploaded_file.name)

                st.cache_resource.clear()

            st.success("✅ Added to Knowledge Base!")
            st.rerun()

    # Show added PDFs
    if st.session_state.uploaded_pdf_names:
        st.markdown("**In knowledge base:**")
        for name in st.session_state.uploaded_pdf_names:
            st.markdown(f"📄 `{name}`")

    st.markdown("---")

    # ── Suggested Questions ──
    st.markdown("### 💡 Suggested Questions")

    suggestions = [
        "Summarize chapter",
        "Explain key concepts",
        "Important formulas",
        "Create short notes",
        "Explain this topic simply",
        "Generate exam questions",
    ]

    for s in suggestions:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state.suggested_question = s
            st.rerun()

    st.markdown("---")

    # ── Clear Chat ──
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.suggested_question = None
        st.rerun()


# =========================================
# MAIN — CHAT AREA
# =========================================
st.markdown("""
<div class="chat-title">
    <span style="font-size:24px">🤖</span>
    <div>
        <h2>AI RAG Assistant</h2>
        <span>Ask questions from your PDFs using AI</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================
# RENDER CHAT HISTORY
# =========================================
def render_messages():
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🔍</div>
            <p>Upload a PDF and start asking questions.<br>Your answers will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for chat in st.session_state.chat_history:

        # User bubble
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble">{chat["question"]}</div>
            <div class="av">U</div>
        </div>
        """, unsafe_allow_html=True)

        # Source chips — deduplicated
        sources_html = ""
        if chat.get("sources"):
            seen = []
            for src in chat["sources"]:
                if src not in seen:
                    seen.append(src)
            chips = "".join(
                f'<span class="src-chip">📄 {src}</span>'
                for src in seen
            )
            sources_html = f'<div class="sources">{chips}</div>'

        # Convert markdown → HTML for answer
        answer_html = md_to_html(chat["answer"])

        st.markdown(f"""
        <div class="msg-ai">
            <div class="av">🤖</div>
            <div class="bubble">
                {answer_html}
                {sources_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


render_messages()


# =========================================
# RETRIEVER
# =========================================
if vectorstore is not None:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
else:
    retriever = None


# =========================================
# SUGGESTED QUESTION PREFILL
# =========================================
prefill = st.session_state.get("suggested_question")
if prefill:
    st.session_state.suggested_question = None


# =========================================
# CHAT INPUT
# =========================================
if retriever is None:
    st.warning("⚠️ Please upload and process a PDF first to start chatting.")
    user_query = None
else:
    user_query = st.chat_input("Ask a question from your PDFs...")
    if prefill and not user_query:
        user_query = prefill


# =========================================
# PROCESS QUERY
# =========================================
if user_query and retriever:

    # Show user message
    st.markdown(f"""
    <div class="msg-user">
        <div class="bubble">{user_query}</div>
        <div class="av">U</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🤔 Thinking..."):

        docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build deduplicated source list using real filenames
        seen_sources = []
        for doc in docs:
            raw = doc.metadata.get("source", "Document")
            fname = os.path.basename(raw)  # fallback clean
            page  = doc.metadata.get("page", "?")
            label = f"{fname} — Page {int(page) + 1}"
            if label not in seen_sources:
                seen_sources.append(label)

        final_prompt = prompt_template.invoke({
            "context": context,
            "question": user_query
        })
        response = llm.invoke(final_prompt)
        answer = response.content

    # Build sources HTML
    sources_html = ""
    if seen_sources:
        chips = "".join(
            f'<span class="src-chip">📄 {src}</span>'
            for src in seen_sources
        )
        sources_html = f'<div class="sources">{chips}</div>'

    # Streaming word-by-word effect (plain text during stream)
    ai_placeholder = st.empty()
    displayed = ""

    for word in answer.split():
        displayed += word + " "
        ai_placeholder.markdown(f"""
        <div class="msg-ai">
            <div class="av">🤖</div>
            <div class="bubble"><p>{displayed}▌</p></div>
        </div>
        """, unsafe_allow_html=True)

    # Final render — convert markdown properly
    answer_html = md_to_html(answer)
    ai_placeholder.markdown(f"""
    <div class="msg-ai">
        <div class="av">🤖</div>
        <div class="bubble">
            {answer_html}
            {sources_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Save to history
    st.session_state.chat_history.append({
        "question": user_query,
        "answer": answer,
        "sources": seen_sources
    })