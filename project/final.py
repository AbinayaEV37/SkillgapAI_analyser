# final.py
"""
Unified SkillGap AI Analyzer - Final integrated Streamlit app
Integrates: module1.py (Milestone1), milestone2/ (Milestone2), module3.py (Milestone3)
Run: streamlit run final.py
"""

import os
import sys
import traceback
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

# -------------------------
# PATHS: ensure milestone2 folder is importable
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
M2_DIR = os.path.join(BASE_DIR, "milestone2")
if M2_DIR not in sys.path:
    sys.path.append(M2_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# -------------------------
# IMPORTS (safe/optional)
# -------------------------
# Milestone 1 (module1.py) - use internal helpers (DocumentUploader, TextExtractor, TextCleaner, process_documents)
try:
    import module1 as m1
    DocumentUploader = m1.DocumentUploader
    TextExtractor = m1.TextExtractor
    TextCleaner = m1.TextCleaner
    process_documents = m1.process_documents
except Exception as e:
    DocumentUploader = None
    TextExtractor = None
    TextCleaner = None
    process_documents = None
    st.warning(f"Milestone1 import failed: {e}")

# Milestone 2 (folder) - core classes
try:
    from milestone2 import text_preprocessor as TextPreprocessor
    #from milestone2 import skill_database as SkillDatabase
    from milestone2.skill_extractor import SkillExtractor
    from milestone2 import bert_embedder as M2SentenceBERTEmbedder
    from milestone2.annotation_ui import AnnotationInterface
    annot = AnnotationInterface()
    from milestone2 import exporter as M2Exporter
    from milestone2 import visualizer as M2SkillVisualizer
    from milestone2 import ner_trainer as CustomNERTrainer
    from milestone2.skill_database import SkillDatabase
    sd = SkillDatabase()



except Exception as e:
    # degrade gracefully but still allow app to run
    TextPreprocessor = None
    SkillDatabase = None
    M2SkillExtractor = None
    M2SentenceBERTEmbedder = None
    AnnotationInterface = None
    M2Exporter = None
    M2SkillVisualizer = None
    CustomNERTrainer = None
    st.warning(f"Milestone2 import issue: {e}")

# Milestone 3 (module3.py) - encoder, gap engine, export manager, learningpath, visuals
try:
    # module3 defines many useful classes: SentenceBERTEncoder, SkillExtractor (name clash), SkillGapEngine,
    # LearningPathGenerator, ExportManager, Visuals, SkillGapApp
    import module3 as m3
    SentenceBERTEncoder = m3.SentenceBERTEncoder
    M3SkillExtractor = m3.SkillExtractor
    SkillGapEngine = m3.SkillGapEngine
    LearningPathGenerator = m3.LearningPathGenerator
    ExportManager = m3.ExportManager
    M3Visuals = m3.Visuals
    Module3SkillGapApp = m3.SkillGapApp
except Exception as e:
    SentenceBERTEncoder = None
    M3SkillExtractor = None
    SkillGapEngine = None
    LearningPathGenerator = None
    ExportManager = None
    M3Visuals = None
    Module3SkillGapApp = None
    st.warning(f"Milestone3 import issue: {e}")

# -------------------------
# Page config and consistent UI style
# -------------------------
st.set_page_config(page_title="SkillGap AI Analyzer", layout="wide", page_icon="🧠")

# ------------------------
# Custom Professional Theme CSS
# ------------------------
st.markdown("""
    <style>
    /* -------- Base Layout -------- */
    .stApp {
        background: linear-gradient(135deg, #e8f0ff 0%, #f9fcff 100%);
        color: #1a1a1a;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* -------- Sidebar -------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12213a 0%, #1e2f4a 100%) !important;
        color: white !important;
        border-right: 1px solid #2a3d61;
        box-shadow: 2px 0 8px rgba(0,0,0,0.1);
    }

    /* Sidebar title & headings */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }

    /* Workflow steps (radio buttons) */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    section[data-testid="stSidebar"] div[role="radiogroup"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0 0 2px rgba(255,255,255,0.4);
        transition: color 0.3s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p,
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover span {
        color: #a8dcff !important;
    }


    /* -------- Main Content Area -------- */
    .main {
        background-color: transparent !important;
        color: #1a1a1a;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #1a2d4d !important;
        font-weight: 600;
    }

    /* -------- Input Elements -------- */
    textarea, input, select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d3dce6 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }
    textarea:focus, input:focus, select:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 8px rgba(74,144,226,0.3);
    }

    /* -------- Buttons -------- */
    button[kind="primary"], button {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: 0.2s ease;
    }
    button:hover {
        background: linear-gradient(90deg, #0061e6, #00a8e6);
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    /* -------- Cards & Containers -------- */
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox {
        border-radius: 10px !important;
        background-color: rgba(255,255,255,0.7) !important;
        padding: 10px;
    }

    /* -------- Scrollbars -------- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #b0c4de;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #8cb3e0;
    }
    /* Remove unwanted blank boxes/spacers in sidebar */
    section[data-testid="stSidebar"] div:empty {
        display: none !important;
        visibility: hidden !important;
}
    </style>
""", unsafe_allow_html=True)


st.title("🧠 SkillGap AI Analyzer")
st.caption("Upload → Parse → Extract → Encode → Compare → Visualize → Export")

# -------------------------
# Sidebar: main navigation (keeps single consistent style)
# -------------------------
st.sidebar.header("Workflow")
step = st.sidebar.radio(
    "Select step",
    (
        "1 - Data Ingestion & Parsing",
        "2 - Skill Extraction (NLP)",
        "3 - Encoding & Similarity",
        "4 - Gap Ranking & Visualize",
        "5 - Annotate / Train NER",
        "6 - Export & Reports",
        "Diagnostics"
    ),
)

# Provide model choices
st.sidebar.markdown("---")
st.sidebar.markdown("**Embedding model (for semantic matching)**")
model_choice = st.sidebar.selectbox("SBERT model", [ "all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2" ], index=0)

# initialize or reuse encoder instances in session state
if "sbert_encoder" not in st.session_state:
    try:
        if SentenceBERTEncoder:
            st.session_state.sbert_encoder = SentenceBERTEncoder(model_choice)
        elif M2SentenceBERTEmbedder:
            # fallback to milestone2 embedder
            st.session_state.sbert_encoder = M2SentenceBERTEmbedder(model_choice)
        else:
            st.session_state.sbert_encoder = None
    except Exception as e:
        st.session_state.sbert_encoder = None
        st.warning(f"Embedding init failed: {e}")

# helper function to reset top-level results
def clear_results():
    for k in ["uploaded_files","processed_docs","extracted_resume_skills","extracted_jd_skills","analysis_df","sim_matrix","overall_match","recs_df"]:
        if k in st.session_state:
            del st.session_state[k]

# -------------------------
# Step 1: Data ingestion & parsing (uses module1 internals if available)
# -------------------------
if step.startswith("1"):
    st.subheader("1 — Data Ingestion & Parsing")
    st.markdown("<div class='muted'>Upload resumes & job descriptions (.pdf, .docx, .txt). Parsing, OCR placeholder, and cleaning applied.</div>", unsafe_allow_html=True)

    # Use DocumentUploader UI (from module1) if available, otherwise fallback
    if DocumentUploader is not None:
        uploader = DocumentUploader()
        uploaded = uploader.upload_files()  # returns list of file dicts when used inside its own UI; but if used here, it will render uploaders again
        # NOTE: module1.DocumentUploader.upload_files expects to run its own UI; to avoid duplicate controls we offer simpler approach:
        # If uploader.upload_files returned files, save them; else provide alternative uploader
        if uploaded:
            st.session_state.uploaded_files = uploaded
    else:
        st.info("DocumentUploader (module1) not found; using generic uploader.")
        resumes = st.file_uploader("Upload Resumes (multiple)", accept_multiple_files=True, type=["pdf","docx","txt"])
        jobs = st.file_uploader("Upload Job Descriptions (multiple)", accept_multiple_files=True, type=["pdf","docx","txt"])
        combined = []
        for f in (resumes or []):
            combined.append({'name': f.name, 'type': 'resume', 'content': f.read(), 'format': f.name.split('.')[-1]})
        for f in (jobs or []):
            combined.append({'name': f.name, 'type': 'job_description', 'content': f.read(), 'format': f.name.split('.')[-1]})
        if combined:
            st.session_state.uploaded_files = combined
            st.success(f"Uploaded {len(combined)} files")

    # Process documents using module1.process_documents if available
    if st.session_state.get("uploaded_files"):
        st.markdown("**Uploaded files:**")
        for f in st.session_state["uploaded_files"]:
            st.write(f"- {f['name']} ({f.get('format', '')}) — type: {f.get('type','')}")
        if st.button("Process / Parse documents", key="process_docs"):
            if process_documents:
                try:
                    with st.spinner("Processing documents..."):
                        docs = process_documents(st.session_state["uploaded_files"])
                        st.session_state.processed_docs = docs
                        st.success(f"Processed {len(docs)} documents")
                except Exception as e:
                    st.error(f"Processing failed: {e}\n{traceback.format_exc()}")
            else:
                st.error("Document processing function not available (module1 import issue).")
    else:
        st.info("No uploaded files yet. Use the upload control above or go to Skill Extraction and paste text.")

    # Preview raw vs cleaned
    if st.session_state.get("processed_docs"):
        st.markdown("### Document preview")
        doc_select = st.selectbox("Choose document to preview", [d['file_name'] for d in st.session_state["processed_docs"]])
        doc = next((d for d in st.session_state["processed_docs"] if d['file_name'] == doc_select), None)
        if doc:
            view_option = st.radio("View", ["Raw text", "Cleaned text"], horizontal=True)
            if view_option == "Raw text":
                st.text_area("Raw text", value=doc.get("raw_text","")[:10000], height=300)
            else:
                st.text_area("Cleaned text", value=doc.get("cleaned_text","")[:10000], height=300)

    st.markdown("---")
    st.button("Clear data & restart", on_click=clear_results)

# -------------------------
# Step 2: Skill extraction (use milestone2 extractor classes)
# -------------------------
elif step.startswith("2"):
    st.subheader("2 — Skill Extraction (NLP)")
    st.markdown("<div class='muted'>Use spaCy/custom NER/keyword matching to extract candidate skills from parsed text.</div>", unsafe_allow_html=True)

    # Allow user to paste text (resume or JD) OR pick from processed docs
    source = st.radio("Input source", ["Use parsed documents", "Paste text (ad-hoc)"], index=0)
    text_to_process = ""
    doc_type = st.selectbox("Document Type", ["resume", "job_description"])
    if source == "Use parsed documents" and st.session_state.get("processed_docs"):
        doc_names = [d['file_name'] for d in st.session_state["processed_docs"]]
        chosen = st.selectbox("Choose parsed document", doc_names)
        doc = next((d for d in st.session_state["processed_docs"] if d['file_name']==chosen), None)
        if doc:
            text_to_process = doc.get("cleaned_text","")
            st.text_area("Preview cleaned text", value=text_to_process[:8000], height=220)
    else:
        text_to_process = st.text_area("Paste resume or job description text here", height=220)

    method = st.selectbox("Extraction method", ["Keyword matching (fast)", "spaCy noun-chunks (best if installed)", "Custom combined extractor"])
    confidence_filter = st.slider("Minimum confidence (if supported)", 0.0, 1.0, 0.0, 0.05)

    if st.button("Extract skills"):
        if not text_to_process or not text_to_process.strip():
            st.error("Please provide text (paste or parsed doc).")
        else:
            # Use milestone2 TextPreprocessor and SkillExtractor if available; fallback to simple keyword spotting from skill DB
            results_resume = []
            try:
                if method == "Keyword matching (fast)":
                    # basic skill DB check
                    if SkillDatabase:
                        sd = SkillDatabase()
                        all_sk = sd.get_all_skills()
                        found = [s for s in all_sk if s.lower() in text_to_process.lower()]
                        results_resume = sorted(set(found))
                    else:
                        # fallback: simple split
                        items = [t.strip() for t in text_to_process.split(",") if t.strip()]
                        results_resume = items[:200]
                elif method == "spaCy noun-chunks (best if installed)":
                    if TextPreprocessor:
                        tp = TextPreprocessor()
                        pre = tp.preprocess_text(text_to_process)
                        if pre.get("success"):
                            noun_chunks = pre.get("noun_chunks", [])
                            # filter short chunks and return
                            results_resume = [c for c in noun_chunks if len(c.split()) <= 4][:500]
                        else:
                            results_resume = []
                    else:
                        results_resume = []
                else:
                    # custom combined extractor from milestone2 (SkillExtractor)
                    if SkillExtractor:
                        extractor = SkillExtractor()

                        out = extractor.extract_skills(text_to_process, document_type=doc_type)
                        if out.get("success"):
                            results_resume = out.get("all_skills", [])
                        else:
                            results_resume = []
                    else:
                        results_resume = []
            except Exception as e:
                st.error(f"Extraction error: {e}\n{traceback.format_exc()}")
                results_resume = []

            # store and display
            st.session_state["extracted_text"] = text_to_process
            st.session_state["extracted_results"] = {
                "skills": results_resume,
                "method": method,
                "source_type": doc_type
            }
            st.success(f"Extracted {len(results_resume)} candidate skills")
            if results_resume:
                # show grouped by simple categories if skill DB exists
                if SkillDatabase:
                    sd = SkillDatabase()
                    cat_map = {}
                    for s in results_resume:
                        cat = sd.get_category_for_skill(s)
                        cat_map.setdefault(cat, []).append(s)
                    for cat, items in cat_map.items():
                        st.markdown(f"**{cat.replace('_',' ').title()} ({len(items)})**")
                        st.write(", ".join(items[:100]))
                else:
                    st.write(results_resume[:200])

    # Save extracted as resume/jd skills for next steps
    if st.session_state.get("extracted_results"):
        if st.button("Save extracted as current resume/jd skills"):
            sks = st.session_state["extracted_results"]["skills"]
            if doc_type == "resume":
                st.session_state["extracted_resume_skills"] = sks
            else:
                st.session_state["extracted_jd_skills"] = sks
            st.success("Saved extracted skills to session")
        # ✅ Sync extracted skills for next steps
        if "extracted_resume_skills" in st.session_state:
            st.session_state["resume_skills"] = st.session_state["extracted_resume_skills"]
        if "extracted_jd_skills" in st.session_state:
            st.session_state["jd_skills"] = st.session_state["extracted_jd_skills"]

# -------------------------
# Step 3: Encoding & Similarity prep
# -------------------------
elif step.startswith("3"):
    st.subheader("3 — Encode Skills & Compute Similarities")
    st.markdown("<div class='muted'>Generate embeddings (SBERT) and prepare similarity matrix for matching.</div>", unsafe_allow_html=True)

    # Get lists from earlier steps or allow manual paste of comma-separated skills
    resume_skills = st.session_state.get("extracted_resume_skills", [])
    jd_skills = st.session_state.get("extracted_jd_skills", [])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Resume skills (session)**")
        st.write(resume_skills[:200])
        if st.button("Paste resume skills manually"):
            inp = st.text_area("Comma-separated resume skills", value=",".join(resume_skills))
            if inp:
                st.session_state["extracted_resume_skills"] = [x.strip() for x in inp.split(",") if x.strip()]
    with col2:
        st.markdown("**Job Description skills (session)**")
        st.write(jd_skills[:200])
        if st.button("Paste JD skills manually"):
            inp = st.text_area("Comma-separated JD skills", value=",".join(jd_skills))
            if inp:
                st.session_state["extracted_jd_skills"] = [x.strip() for x in inp.split(",") if x.strip()]

    # Re-read after potential manual entry
    resume_skills = st.session_state.get("extracted_resume_skills", [])
    jd_skills = st.session_state.get("extracted_jd_skills", [])

    st.markdown("**Embedding settings**")
    use_cache = st.checkbox("Use embedding cache (file)", value=True)
    sim_threshold = st.slider("Similarity threshold (to consider 'covered')", 0.0, 1.0, 0.60, 0.01)
    st.session_state["sim_threshold"] = sim_threshold


    if st.button("Generate embeddings & compute similarity"):
        if not resume_skills or not jd_skills:
            st.error("Provide both resume and JD skills first (from extraction step).")
        else:
            try:
                # create encoder (prefer module3 encoder for caching)
                encoder = st.session_state.get("sbert_encoder")
                if encoder is None:
                    if SentenceBERTEncoder:
                        encoder = SentenceBERTEncoder(model_choice)
                        st.session_state.sbert_encoder = encoder
                    elif M2SentenceBERTEmbedder:
                        encoder = M2SentenceBERTEmbedder(model_choice)
                        st.session_state.sbert_encoder = encoder
                    else:
                        encoder = None

                if encoder is None:
                    st.warning("No SBERT encoder available; computing TF-IDF fallback embeddings (less accurate).")

                # encode via encoder interface - module3 encoder returns numpy arrays
                if hasattr(encoder, "encode_skills"):
                    emb_resume = encoder.encode_skills(resume_skills, use_cache=use_cache)
                    emb_jd = encoder.encode_skills(jd_skills, use_cache=use_cache)
                else:
                    # fallback: compute simple TF-IDF vectors using sklearn inside-app (cheap)
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    vect = TfidfVectorizer(max_features=256)
                    emb_resume = vect.fit_transform(resume_skills).toarray()
                    emb_jd = vect.transform(jd_skills).toarray()

                # compute similarity matrix (jd x resume)
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(emb_jd, emb_resume)
                st.session_state["sim_matrix"] = sim
                st.session_state["resume_u"] = resume_skills
                st.session_state["jd_u"] = jd_skills
                st.success("Similarity matrix computed and cached in session")
            except Exception as e:
                st.error(f"Embedding/similarity computation failed: {e}\n{traceback.format_exc()}")

# -------------------------
# Step 4: Gap ranking & Visualize (uses module3 SkillGapEngine & Visuals if available)
# -------------------------
elif step.startswith("4"):
    st.subheader("4 — Gap Ranking & Visualizations")
    st.markdown("<div class='muted'>Rank missing skills by importance × gap, view heatmap, radar and missing priority list.</div>", unsafe_allow_html=True)
    sim_threshold = st.session_state.get("sim_threshold", 0.6)
    # --- Safety check for missing skills ---
    resume_skills = st.session_state.get("extracted_resume_skills", [])
    jd_skills = st.session_state.get("extracted_jd_skills", [])

    if not resume_skills or not jd_skills:
        st.warning("⚠️ Please make sure both Resume and Job Description skills are extracted and saved in Step 2 before continuing.")
        st.stop()


    df = st.session_state.get("analysis_df")
    sim = st.session_state.get("sim_matrix")
    resume_u = st.session_state.get("resume_u", [])
    jd_u = st.session_state.get("jd_u", [])
    overall = st.session_state.get("overall_match", 0.0)

    if st.button("Run gap ranking (engine)"):
        # Use module3.SkillGapEngine if available to produce DataFrame with gap scores
        resume_skills = resume_u
        jd_skills = jd_u
        if not resume_skills or not jd_skills:
            st.error("Please compute embeddings & similarity first (Step 3).")
        else:
            try:
                if SkillGapEngine and st.session_state.get("sbert_encoder"):
                    engine = SkillGapEngine(st.session_state.sbert_encoder, threshold=st.session_state.get("sim_threshold", 0.6) or 0.6)
                    df_res, sim_matrix, res_list, jd_list, overall_weighted = engine.compute(resume_skills, jd_skills)
                    st.session_state["analysis_df"] = df_res
                    st.session_state["sim_matrix"] = sim_matrix
                    st.session_state["resume_u"] = res_list
                    st.session_state["jd_u"] = jd_list
                    st.session_state["overall_match"] = overall_weighted
                    st.success(f"Gap analysis complete — overall match: {overall_weighted:.2f}%")
                else:
                    # fallback simpler ranking using sim matrix from session
                    if sim is None or sim.size == 0:
                        st.error("No similarity matrix available to compute gaps.")
                    else:
                        rows = []
                        # simple importance = 1 for all
                        import numpy as np
                        for j_idx, j_skill in enumerate(jd_skills):
                            col = sim[j_idx, :] if sim.size else np.array([])
                            if col.size:
                                best_i = int(np.argmax(col))
                                best_score = float(col[best_i])
                                best_resume = resume_skills[best_i]
                            else:
                                best_score = 0.0
                                best_resume = None
                            status = "Covered" if best_score >= sim_threshold else "Missing"
                            importance = 1.0
                            gap = round(importance * (1.0 - best_score), 4)
                            rows.append({"JD Skill": j_skill, "Best Resume Match": best_resume, "Similarity": round(best_score,4), "Status": status, "Importance": importance, "gap_score": gap})
                        df_s = pd.DataFrame(rows).sort_values("gap_score", ascending=False).reset_index(drop=True)
                        st.session_state["analysis_df"] = df_s
                        st.success("Computed fallback gap ranking")
            except Exception as e:
                st.error(f"Gap engine failed: {e}\n{traceback.format_exc()}")

    # Display analysis results and visuals
    df = st.session_state.get("analysis_df", pd.DataFrame())
    if not df.empty:
        st.markdown("### Summary")
        overall = st.session_state.get("overall_match", 0.0)
        st.metric("Overall weighted match (%)", f"{overall:.2f}%")
        st.markdown("### Detailed matches")
        st.dataframe(df, use_container_width=True)

        # Heatmap
        if st.session_state.get("sim_matrix") is not None:
            if M3Visuals:
                vis = M3Visuals()
                st.markdown("### Similarity heatmap (JD x Resume)")
                vis.plot_heatmap(st.session_state.get("resume_u", []), st.session_state.get("jd_u", []), st.session_state.get("sim_matrix"))
            else:
                st.info("Advanced visuals not available (module3). Using basic heatmap.")
                import plotly.express as px
                sim_small = st.session_state.get("sim_matrix")
                if sim_small is not None and sim_small.size:
                    try:
                        dfhm = pd.DataFrame(sim_small, index=st.session_state.get("jd_u", []), columns=st.session_state.get("resume_u", []))
                        st.dataframe(dfhm.iloc[:40, :40])
                    except Exception:
                        st.write("Heatmap preview not available due to size.")

        # Missing bars & radar if available
        if M3Visuals:
            vis.plot_missing_bar(df, top_n=12)
            vis.plot_radar(df)
        else:
            st.markdown("Top missing skills:")
            missing = df[df["Status"]=="Missing"].sort_values("gap_score", ascending=False).head(10)
            for _, r in missing.iterrows():
                st.write(f"- {r['JD Skill']} (gap_score: {r['gap_score']:.3f})")
        # ----------------------------------------------------------------
        # Integrated Learning Recommendations (from module3)
        # ----------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🎯 Personalized Learning Recommendations")
        st.markdown("<div class='muted'>AI-suggested courses for the skills identified as missing.</div>", unsafe_allow_html=True)

        # Derive missing skills dynamically from current DataFrame
        df_results = st.session_state.get("analysis_df", pd.DataFrame())
        if not df_results.empty:
            missing_skills = df_results[df_results["Status"] == "Missing"]["JD Skill"].tolist()
            st.session_state["missing_skills"] = missing_skills
        else:
            missing_skills = []

        if not missing_skills:
            st.info("No missing skills found — please complete skill gap analysis first.")
        else:
            st.success(f"Detected {len(missing_skills)} missing skills. Generating learning recommendations...")

            try:
                if LearningPathGenerator:
                    lp_gen = LearningPathGenerator()
                    rec_df = lp_gen.recommend(missing_skills)

                    if rec_df is not None and not rec_df.empty:
                        st.session_state["learning_recommendations"] = rec_df

                        # Show as table directly here (no separate UI call)
                        st.markdown("#### 📘 Recommended Learning Paths")
                        st.dataframe(rec_df, use_container_width=True)

                        # Optional export button
                        csv = rec_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Learning Path Report (CSV)",
                            data=csv,
                            file_name="learning_path_recommendations.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("No valid recommendations found for these skills.")
                else:
                    st.warning("⚠️ LearningPathGenerator module not loaded. Please check module3 import.")
            except Exception as e:
                st.error(f"Failed to generate learning recommendations: {e}")


# -------------------------
# Step 5: Annotation / Train Custom NER (milestone2.annotation_ui & ner_trainer)
# -------------------------
elif step.startswith("5"):
    st.subheader("5 — Annotation Interface & Train Custom NER")
    st.markdown("<div class='muted'>Create labeled skill annotations and optionally train a custom spaCy NER model.</div>", unsafe_allow_html=True)

    if AnnotationInterface:
        try:
            annot = AnnotationInterface()
            annot.create_annotation_ui()
        except Exception as e:
            st.error(f"Annotation UI error: {e}\n{traceback.format_exc()}")
    else:
        st.warning("Annotation UI module not available (ensure milestone2/annotation_ui.py exists).")

    st.markdown("---")
    if CustomNERTrainer:
        st.markdown("**NER Trainer**")
        if st.button("Create blank custom NER model"):
            try:
                trainer = CustomNERTrainer()
                trainer.create_blank_model()
                st.session_state["custom_ner"] = trainer
                st.success("Blank NER model created in session.")
            except Exception as e:
                st.error(f"Failed to create blank NER: {e}")
        trainer_obj = st.session_state.get("custom_ner")
        if trainer_obj:
            st.write("Trainer available in session.")
            if st.button("Train from saved annotations (session)"):
                ann = st.session_state.get("training_annotations", [])
                if ann:
                    try:
                        spacy_data = trainer_obj.prepare_training_data(ann)
                        stats = trainer_obj.train(spacy_data, n_iter=20)
                        st.session_state["trained_ner"] = trainer_obj
                        st.success("Training finished.")
                        st.write(stats)
                    except Exception as e:
                        st.error(f"Training error: {e}\n{traceback.format_exc()}")
                else:
                    st.warning("No annotations found in session. Use the Annotation Interface to create/save annotations.")
    else:
        st.info("Custom NER trainer not available (milestone2.ner_trainer.py missing).")

# -------------------------
# Step 6: Export & Reports
# -------------------------
elif step.startswith("6"):
    st.subheader("6 — Export & Reports")
    st.markdown("<div class='muted'>Export analysis results as CSV, JSON, and a styled PDF report.</div>", unsafe_allow_html=True)

    df = st.session_state.get("analysis_df", pd.DataFrame())
    overall = st.session_state.get("overall_match", 0.0)

    if df is None or df.empty:
        st.warning("No analysis results found. Run Step 4 to compute gap analysis first.")
    else:
        st.markdown("### Export options")
        # Use module3 ExportManager if available (nicely formatted PDF), else use milestone2 Exporter as fallback
        try:
            if ExportManager:
                manager = ExportManager(df, {"Overall Match": f"{overall:.2f}%"}, title="SkillGap AI Analyzer Report")
                csv_bytes = manager.export_csv()
                pdf_bytes = manager.export_pdf()
                st.download_button("⬇️ Download CSV", csv_bytes, file_name="skillgap_report.csv", mime="text/csv")
                st.download_button("📘 Download PDF (stylized)", pdf_bytes, file_name="skillgap_report.pdf", mime="application/pdf")
            elif M2Exporter:
                # M2Exporter expects result dict; adapt:
                result = {
                    "statistics": {"total_skills": len(df)},
                    "all_skills": df["JD Skill"].tolist() if "JD Skill" in df.columns else [],
                    "categorized_skills": {},
                    "skill_confidence": {}
                }
                csv_str = M2Exporter.to_csv(result)
                json_str = M2Exporter.to_json(result)
                text_report = M2Exporter.to_text_report(result)
                st.download_button("⬇️ Download CSV", csv_str, file_name="skillgap_report.csv", mime="text/csv")
                st.download_button("📘 Download JSON", json_str, file_name="skillgap_report.json", mime="application/json")
                st.download_button("📄 Download Text Report", text_report, file_name="skillgap_report.txt", mime="text/plain")
            else:
                # fallback simple CSV via pandas
                st.download_button("⬇️ Download CSV (fallback)", df.to_csv(index=False).encode("utf-8"), file_name="skillgap_simple.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Export failed: {e}\n{traceback.format_exc()}")

# -------------------------
# Diagnostics / Misc
# -------------------------
else:
    st.subheader("Diagnostics")
    st.markdown("<div class='muted'>Engine status, available models, caches and simple logs.</div>", unsafe_allow_html=True)
    info = {
        "module1_loaded": bool(DocumentUploader and process_documents),
        "milestone2_loaded": bool(TextPreprocessor and SkillExtractor),
        "module3_loaded": bool(SentenceBERTEncoder and SkillGapEngine),
        "sbert_in_session": bool(st.session_state.get("sbert_encoder") is not None),
        "processed_docs": len(st.session_state.get("processed_docs", []))
    }
    st.json(info)

    if st.button("Clear session state (keep code)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
