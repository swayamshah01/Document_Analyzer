"""
app.py
------
Universal Document Intelligence Tool — Main Streamlit Application.

Entry point: `streamlit run app.py`

Architecture:
    Sidebar  → model config, file upload, entity filter controls
    Tab 1    → Entity Analysis  (displaCy NER + entity table + download)
    Tab 2    → Semantic Search  (clause finder with cosine similarity scores)
    Tab 3    → Raw Text         (debugging expander)
"""

import sys
import time
import numpy as np
import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Document Intelligence Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, ".")          # ensure src/ is importable when run from project root
from src.extractor import extract_text_from_pdf, compute_text_stats
from src.nlp_pipeline import (
    load_spacy_model,
    process_document,
    filter_business_entities,
    entities_to_dataframe,
    get_entity_summary,
    BUSINESS_LABELS,
    LABEL_DESCRIPTIONS,
    DISPLACY_COLORS,
)
from src.similarity import (
    load_similarity_model,
    split_into_segments,
    encode_segments,
    find_similar_segments,
)
from src.utils import (
    df_to_csv_bytes,
    df_to_excel_bytes,
    truncate_text,
    highlight_query_in_segment,
)




# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — Navy / Teal palette only
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Fonts: Manrope (Display) & Inter (Body) ──────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #1a1b1e !important;
}

h1, h2, h3, h4, h5, h6, .metric-value, .hero-header h1 {
    font-family: 'Manrope', sans-serif !important;
    color: #001b3d !important;
    letter-spacing: -0.02em;
}

/* ── Main background ─────────────────────────────────────── */
.stApp {
    background: #faf9fd !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1B365D !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] .stMarkdown p, 
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] label {
    color: #d1d9e6 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

/* ── Hero header ─────────────────────────────────────────── */
.hero-header {
    background: rgba(250, 249, 253, 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 0;
    padding: 32px 0 40px;
    margin-bottom: 24px;
    border-bottom: none;
}
.hero-header h1 {
    margin: 0;
    font-size: 3.5rem; /* display-lg */
    font-weight: 700;
    color: #001b3d !important;
    letter-spacing: -0.02em;
}
.hero-header p {
    margin: 12px 0 0;
    font-size: 1.1rem;
    color: #586377 !important;
    font-weight: 400;
}

/* ── Metric cards ────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 24px;
    text-align: left;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
    margin-bottom: 24px;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #002046 !important;
    line-height: 1.1;
    font-family: 'Manrope', sans-serif !important;
}
.metric-label {
    font-size: 0.85rem;
    color: #586377 !important;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
    font-family: 'Inter', sans-serif !important;
}

/* ── Section cards ───────────────────────────────────────── */
.section-card {
    background: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 32px;
    margin-bottom: 32px;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background: transparent;
    border-radius: 0;
    padding: 0;
    border: none;
    border-bottom: 1px solid rgba(26, 27, 30, 0.15); /* Ghost Border */
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    color: #586377 !important;
    font-weight: 500;
    padding: 12px 0;
    background: transparent !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    box-shadow: none !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #002046 !important;
    font-weight: 600;
    border-bottom: 2px solid #002046 !important;
}

/* ── Similarity result cards ─────────────────────────────── */
.sim-card {
    background: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
    position: relative;
}
/* No Line Rule */
.sim-score {
    display: inline-block;
    background: #f4f3f7 !important;
    color: #002046 !important;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 16px;
    font-family: 'Manrope', sans-serif !important;
}
.sim-rank {
    font-size: 0.75rem;
    color: #586377 !important;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.sim-card p {
    color: #1a1b1e !important;
    line-height: 1.6;
}

/* Highlight Overlays for OCR/Similarity */
.sim-card mark {
    background-color: rgba(214, 227, 255, 0.3) !important; /* primary_fixed at 30% */
    color: #1a1b1e !important;
    padding: 0 2px;
}

/* ── Entity badge pills ──────────────────────────────────── */
.entity-pill {
    display: inline-block;
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
    background: #f4f3f7;
    border: none;
    color: #001b3d;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: #002046 !important;
    color: #ffffff !important;
    border: none;
    border-radius: 0.375rem; /* 6px */
    font-weight: 500;
    padding: 10px 24px;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.1);
}

/* ── Inputs ──────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: #f4f3f7 !important;
    border: none !important;
    border-bottom: 1px solid rgba(26, 27, 30, 0.4) !important;
    border-radius: 4px 4px 0 0 !important;
    color: #1a1b1e !important;
    padding: 12px 16px !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-bottom: 2px solid #002046 !important;
    background: #e9e7eb !important;
}

/* ── DataFrames ──────────────────────────────────────────── */
.dataframe {
    background: #ffffff !important;
    border: 1px solid rgba(26, 27, 30, 0.15) !important; /* Ghost border */
    border-radius: 4px;
}
.dataframe th {
    background: #f4f3f7 !important;
    color: #586377 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Expander ────────────────────────────────────────────── */
details {
    background: #ffffff;
    border: none !important;
    border-radius: 8px;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
}

/* ── Divider ─────────────────────────────────────────────── */
hr {
    border-color: rgba(26, 27, 30, 0.15) !important;
}

/* ── displaCy container ──────────────────────────────────── */
.displacy-container {
    background: #ffffff;
    padding: 32px;
    border-radius: 8px;
    border: none;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
    overflow-x: auto;
    line-height: 2.8;
    font-size: 0.95rem;
    color: #1a1b1e;
}

/* ── Status info boxes ───────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 16px 32px rgba(0, 32, 70, 0.06);
}

/* ── Sidebar logo area ───────────────────────────────────── */
.sidebar-logo {
    text-align: left;
    padding: 24px 0 32px;
    border-bottom: none;
    margin-bottom: 24px;
}
.sidebar-logo .logo-icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 12px;
}
.sidebar-logo h3 {
    color: #ffffff !important;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    font-family: 'Manrope', sans-serif !important;
}
.sidebar-logo p {
    color: #d1d9e6 !important;
    font-size: 0.85rem;
    margin: 4px 0 0;
}

/* ── Displacy Highlights Overlays (OCR/Approved concepts) ── */
.displacy-container mark {
    border-radius: 4px !important;
    padding: 0.25em 0.4em !important;
}

/* ── Dynamic Typography scales ───────────────────────────── */
h2 { font-size: 2rem !important; }
h3 { font-size: 1.5rem !important; }

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "full_text": None,
        "page_texts": [],
        "page_count": 0,
        "doc": None,
        "entities": [],
        "entity_df": pd.DataFrame(),
        "segments": [],
        "segment_embeddings": None,
        "nlp_model_name": "en_core_web_sm",
        "sim_model_name": "all-MiniLM-L6-v2",
        "selected_labels": list(BUSINESS_LABELS),
        "segment_size": 300,
        "overlap": 50,
        "top_k": 5,
        "last_filename": None,
        "processing_done": False,
        "last_segment_size": 300,
        "last_overlap": 50,
        "last_sim_model_name": "all-MiniLM-L6-v2",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-header">
        <h1>📄 Universal Document Analyzer</h1>
        <p>Upload any PDF to extract critical entities, analyze text content, and find semantically similar clauses using AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Configuration Panel ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📂 Upload Document PDF",
    type=["pdf"],
    help="Supports text-selectable PDFs. Scanned/image PDFs require OCR.",
)

with st.expander("⚙️ Analysis Settings & Filters", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**🏷️ Entity Filters**")
        selected_labels = st.multiselect(
            "Select categories to track:",
            options=sorted(BUSINESS_LABELS),
            default=["PERSON", "ORG", "MONEY", "DATE", "PRODUCT", "LOC"],
            format_func=lambda x: f"{x} — {LABEL_DESCRIPTIONS.get(x, x)}",
            label_visibility="collapsed"
        )
        st.session_state["selected_labels"] = selected_labels or list(BUSINESS_LABELS)
        
        st.markdown("**🧠 Processing Model**")
        spacy_model = st.selectbox(
            "NER Model",
            options=["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state["nlp_model_name"] = spacy_model

    with col2:
        st.markdown("**🔍 Semantic Engine**")
        sim_model = st.selectbox(
            "Similarity Model",
            options=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state["sim_model_name"] = sim_model
        
        seg_size = st.slider("Segment size (words)", 100, 600, 300, 50)
        st.session_state["segment_size"] = seg_size
        
        overlap = st.slider("Segment overlap", 0, 150, 50, 10)
        st.session_state["overlap"] = overlap

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS UPLOADED FILE
# ─────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    filename = uploaded_file.name

    # Only reprocess if a new file is uploaded
    if filename != st.session_state["last_filename"]:
        st.session_state["processing_done"] = False
        st.session_state["last_filename"] = filename

    if not st.session_state["processing_done"]:
        progress_bar = st.progress(0, text="🔍 Starting extraction…")

        try:
            # ── Step 1: Extract text ──────────────────────────────────
            progress_bar.progress(10, text="📄 Extracting text from PDF…")
            full_text, page_texts, page_count = extract_text_from_pdf(uploaded_file)
            st.session_state["full_text"] = full_text
            st.session_state["page_texts"] = page_texts
            st.session_state["page_count"] = page_count

            # ── Step 2: Load spaCy ────────────────────────────────────
            progress_bar.progress(30, text="🧠 Loading NER model…")
            nlp = load_spacy_model(st.session_state["nlp_model_name"])

            # ── Step 3: NLP processing ────────────────────────────────
            progress_bar.progress(50, text="🔬 Running NLP pipeline…")
            doc = process_document(full_text, nlp)
            st.session_state["doc"] = doc

            # ── Step 4: Entity extraction ──────────────────────────────
            progress_bar.progress(65, text="🏷️ Extracting entities…")
            entities = filter_business_entities(
                doc,
                labels=set(st.session_state["selected_labels"]),
            )
            st.session_state["entities"] = entities
            st.session_state["entity_df"] = entities_to_dataframe(entities)

            # ── Step 5: Segment text for similarity ────────────────────
            progress_bar.progress(75, text="✂️ Segmenting text…")
            segments = split_into_segments(
                full_text,
                segment_size=st.session_state["segment_size"],
                overlap=st.session_state["overlap"],
            )
            st.session_state["segments"] = segments

            # ── Step 6: Load similarity model & embed ─────────────────
            progress_bar.progress(85, text="📐 Loading similarity model & encoding…")
            sim_model_obj = load_similarity_model(st.session_state["sim_model_name"])
            if segments:
                segment_embeddings = encode_segments(segments, sim_model_obj)
                st.session_state["segment_embeddings"] = segment_embeddings
            else:
                st.session_state["segment_embeddings"] = None

            progress_bar.progress(100, text="✅ Done!")
            time.sleep(0.4)
            progress_bar.empty()
            st.session_state["processing_done"] = True
            st.success(f"✅ **{filename}** processed successfully!")

        except (ValueError, RuntimeError, OSError) as exc:
            progress_bar.empty()
            st.error(f"❌ {exc}")
            st.stop()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"❌ Unexpected error: {exc}")
            st.stop()

    # ── Check if segment parameters changed ──────────────────
    if st.session_state["processing_done"] and st.session_state["full_text"]:
        needs_resegment = False
        if st.session_state["segment_size"] != st.session_state.get("last_segment_size"):
            needs_resegment = True
            st.session_state["last_segment_size"] = st.session_state["segment_size"]
        
        if st.session_state["overlap"] != st.session_state.get("last_overlap"):
            needs_resegment = True
            st.session_state["last_overlap"] = st.session_state["overlap"]
            
        if st.session_state["sim_model_name"] != st.session_state.get("last_sim_model_name"):
            needs_resegment = True
            st.session_state["last_sim_model_name"] = st.session_state["sim_model_name"]

        if needs_resegment:
            with st.spinner("✂️ Updating segments and embeddings..."):
                segments = split_into_segments(
                    st.session_state["full_text"],
                    segment_size=st.session_state["segment_size"],
                    overlap=st.session_state["overlap"],
                )
                st.session_state["segments"] = segments
                
                sim_model_obj = load_similarity_model(st.session_state["sim_model_name"])
                if segments:
                    segment_embeddings = encode_segments(segments, sim_model_obj)
                    st.session_state["segment_embeddings"] = segment_embeddings
                else:
                    st.session_state["segment_embeddings"] = None


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD  (only shown after successful processing)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["processing_done"] and st.session_state["full_text"]:
    full_text = st.session_state["full_text"]
    entity_df = st.session_state["entity_df"]
    doc = st.session_state["doc"]
    segments = st.session_state["segments"]

    # ── Stats row ─────────────────────────────────────────────────────────────
    stats = compute_text_stats(full_text, st.session_state["page_count"])

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_data = [
        (c1, stats["pages"], "Pages"),
        (c2, f'{stats["words"]:,}', "Words"),
        (c3, f'{stats["sentences"]:,}', "Sentences"),
        (c4, len(entity_df), "Entities Found"),
        (c5, len(segments), "Text Segments"),
    ]
    for col, val, label in metric_data:
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_entity, tab_similarity, tab_raw = st.tabs(
        ["🏷️  Entity Analysis", "🔍  Semantic Search", "📝  Raw Text"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Entity Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tab_entity:

        if entity_df.empty:
            st.info("No entities found with the current filter selection. Try adding more entity types in the sidebar.")
        else:
            # ── Summary pills ─────────────────────────────────────────
            summary_df = get_entity_summary(entity_df)
            st.markdown("#### Entity Summary")
            pill_html = ""
            for _, row in summary_df.iterrows():
                label = row["Label"]
                count = row["Count"]
                pill_html += (
                    f'<span class="entity-pill label-{label}">'
                    f'{label} &nbsp;<b>{count}</b></span>'
                )
            st.markdown(pill_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── displaCy NER Visualisation ────────────────────────────
            st.markdown("#### 🔬 NER Visualisation")
            st.markdown(
                "*Scroll right if the text is wider than the viewport. "
                "Entities are colour-coded by type.*"
            )

            # Render displaCy HTML for the first ~5 000 chars to keep it snappy
            preview_text = full_text[:5000]
            preview_doc = doc.char_span(0, min(5000, len(full_text)), alignment_mode="expand") or doc

            # Build a mini doc from the slice for displacy
            nlp_obj = load_spacy_model(st.session_state["nlp_model_name"])
            preview_doc2 = nlp_obj(preview_text)

            html = displacy.render(
                preview_doc2,
                style="ent",
                options={
                    "ents": st.session_state["selected_labels"],
                    "colors": DISPLACY_COLORS,
                },
                page=False,
            )
            st.markdown(
                f'<div class="displacy-container">{html}</div>',
                unsafe_allow_html=True,
            )

            if len(full_text) > 5000:
                st.caption(f"📌 Visualisation shows the first 5,000 characters. Full document has {len(full_text):,} characters.")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Full entity table ─────────────────────────────────────
            st.markdown("#### 📋 Extracted Entity Table")

            label_filter = st.multiselect(
                "Filter by label",
                options=sorted(entity_df["Label"].unique()),
                default=sorted(entity_df["Label"].unique()),
                key="entity_table_filter",
            )
            filtered_df = entity_df[entity_df["Label"].isin(label_filter)] if label_filter else entity_df

            st.dataframe(
                filtered_df[["Entity", "Label", "Description"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entity": st.column_config.TextColumn("Entity", width="large"),
                    "Label": st.column_config.TextColumn("Label", width="small"),
                    "Description": st.column_config.TextColumn("Category Description", width="large"),
                },
            )

            # ── Download buttons ──────────────────────────────────────
            st.markdown("#### ⬇️ Export Results")
            dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 3])

            with dl_col1:
                st.download_button(
                    label="📥 Download CSV",
                    data=df_to_csv_bytes(filtered_df),
                    file_name="insurance_entities.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with dl_col2:
                st.download_button(
                    label="📊 Download Excel",
                    data=df_to_excel_bytes(filtered_df, sheet_name="Entities"),
                    file_name="insurance_entities.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Semantic Search
    # ════════════════════════════════════════════════════════════════════════
    with tab_similarity:

        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin:0 0 8px; color:#00B4D8;">🤖 How Semantic Search Works</h4>
                <p style="color:#8B949E;margin:0;font-size:0.88rem;">
                Enter a <b>Master Clause</b> or natural-language query below. The AI converts both your query
                and all document segments into high-dimensional vectors (<i>embeddings</i>) and finds the
                passages whose <b>meaning</b> is closest to yours—even if the exact words differ.
                This is <b>Cosine Similarity</b> over a semantic embedding space.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        query = st.text_area(
            "Master Clause / Search Query",
            placeholder="e.g. What is the maximum liability coverage limit for third-party claims?",
            height=110,
            key="similarity_query",
        )

        search_btn = st.button("🔍 Find Similar Clauses", use_container_width=False)

        if search_btn:
            if not query.strip():
                st.warning("⚠️ Please enter a search query.")
            elif not segments:
                st.error("❌ No document segments available. Please re-upload the PDF.")
            else:
                with st.spinner("🔄 Computing semantic similarity…"):
                    try:
                        sim_model_obj = load_similarity_model(
                            st.session_state["sim_model_name"]
                        )
                        results = find_similar_segments(
                            query=query,
                            segments=segments,
                            model=sim_model_obj,
                            segment_embeddings=st.session_state["segment_embeddings"],
                            top_k=st.session_state["top_k"],
                        )
                        st.session_state["last_search_results"] = results
                        st.session_state["last_search_query"] = query
                    except (ValueError, RuntimeError) as exc:
                        st.error(f"❌ {exc}")

        # Display results from session state so they don't disappear on widget interact
        if st.session_state.get("last_search_results") is not None:
            results = st.session_state["last_search_results"]
            last_query = st.session_state["last_search_query"]
            
            if len(results) == 0:
                st.info(f"No relevant clauses found matching your query.")
            else:
                st.markdown(
                    f"<br>**Top {len(results)} semantically similar passage(s) for:** "
                    f'*"{last_query[:80]}{"…" if len(last_query)>80 else ""}"*',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                
                for result in results:
                    score_pct = result["score_pct"]
                    highlighted = highlight_query_in_segment(
                        result["segment"], last_query
                    )
                    color = (
                        "#2EC4B6" if score_pct >= 70
                        else "#00B4D8" if score_pct >= 50
                        else "#8B949E"
                    )
                    st.markdown(
                        f"""
                        <div class="sim-card">
                            <div class="sim-rank">Result #{result['rank']} · Segment {result['segment_index']+1}</div>
                            <span class="sim-score" style="background:linear-gradient(135deg,{color},{color}99);">
                                {score_pct}% match
                            </span>
                            <p style="color:#E6EDF3;font-size:0.9rem;margin:8px 0 0;line-height:1.7;">
                                {highlighted}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Export similarity results ──────────────────────────
                sim_df = pd.DataFrame(
                    [
                        {
                            "Rank": r["rank"],
                            "Score (%)": r["score_pct"],
                            "Segment": r["segment"],
                        }
                        for r in results
                    ]
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 Download Similarity Results (CSV)",
                    data=df_to_csv_bytes(sim_df),
                    file_name="similarity_results.csv",
                    mime="text/csv",
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Raw Text
    # ════════════════════════════════════════════════════════════════════════
    with tab_raw:

        st.markdown("#### 📄 Extracted & Cleaned Text")
        st.caption(
            "This is the full text after extraction and cleaning. "
            "Use this to verify extraction quality and debug NLP issues."
        )

        view_mode = st.radio(
            "View mode",
            options=["Full document", "Per page"],
            horizontal=True,
            key="raw_view_mode",
        )

        if view_mode == "Full document":
            with st.expander("📃 Full Document Text", expanded=True):
                st.text_area(
                    "Cleaned Text",
                    value=truncate_text(full_text, max_chars=10_000),
                    height=500,
                    key="raw_full_text",
                    label_visibility="collapsed",
                )
                if len(full_text) > 10_000:
                    st.caption(
                        f"Showing first 10,000 of {len(full_text):,} characters."
                    )
        else:
            page_select = st.selectbox(
                "Select page",
                options=list(range(1, st.session_state["page_count"] + 1)),
                format_func=lambda x: f"Page {x}",
                key="raw_page_select",
            )
            page_text = st.session_state["page_texts"][page_select - 1]
            with st.expander(f"📃 Page {page_select} Text", expanded=True):
                st.text_area(
                    f"Page {page_select}",
                    value=page_text if page_text else "(No text extracted from this page)",
                    height=400,
                    key=f"raw_page_{page_select}",
                    label_visibility="collapsed",
                )

else:
    # ── Landing / empty state ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    features = [
        ("🗂️", "PDF Extraction", "Handles multi-column layouts & embedded tables using pdfplumber with regex-based text cleaning."),
        ("🧠", "NER Analysis", "Named Entity Recognition via spaCy en_core_web_sm — filters PERSON, ORG, MONEY, DATE and more."),
        ("🔍", "Semantic Search", "Find clauses by meaning, not keywords, using sentence-transformers cosine similarity."),
    ]
    for col, (icon, title, desc) in zip([col_a, col_b, col_c], features):
        col.markdown(
            f"""
            <div class="section-card" style="text-align:center;padding:32px 24px;">
                <div style="font-size:2.4rem;margin-bottom:14px;">{icon}</div>
                <h4 style="color:#00B4D8;margin:0 0 10px;">{title}</h4>
                <p style="color:#8B949E;font-size:0.88rem;line-height:1.6;margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_d, col_e = st.columns(2)
    features2 = [
        ("📊", "Structured Export", "Download extracted entities to CSV or Excel with one click. Every analysis is audit-ready."),
        ("⬛", "Vector Math", "Embeddings are L2-normalised; similarity = dot product. Powered by all-MiniLM-L6-v2 (80MB model)."),
    ]
    for col, (icon, title, desc) in zip([col_d, col_e], features2):
        col.markdown(
            f"""
            <div class="section-card" style="text-align:center;padding:32px 24px;">
                <div style="font-size:2.4rem;margin-bottom:14px;">{icon}</div>
                <h4 style="color:#00B4D8;margin:0 0 10px;">{title}</h4>
                <p style="color:#8B949E;font-size:0.88rem;line-height:1.6;margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 **Upload an insurance PDF** using the sidebar to get started.")
