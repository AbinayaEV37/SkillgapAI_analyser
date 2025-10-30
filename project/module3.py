# module3.py
# SkillGap AI Analyzer - Milestone 3 
import os
import io
import re
import json
import math
import logging
import tempfile
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fpdf import FPDF

# Optional spaCy import (if available)
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# SentenceTransformers import
from sentence_transformers import SentenceTransformer, util

# ----------------- Logging -----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "milestone3.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("milestone3")
logger.setLevel(logging.INFO)

# ----------------- Constants -----------------
DEFAULT_SBERT = "all-MiniLM-L6-v2"
EMBED_CACHE_FILE = "embed_cache.json"
TFIDF_DIM = 256

# ----------------- Utilities -----------------
def clean_text(s: str, lowercase: bool = True) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if lowercase:
        t = t.lower()
    # replace punctuation with space except +#.- (keep those)
    t = re.sub(r"[^\w\s\+#\.\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similar_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_json_read(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"safe_json_read failed: {e}")
        return {}

def safe_json_write(path: str, obj: Any):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"safe_json_write failed: {e}")

# ----------------- SentenceBERTEncoder (mentor class) -----------------
import typing
class SentenceBERTEncoder:
    """Handles BERT embedding generation using Sentence-BERT (mentor style)."""

    def __init__(self, model_name: str = DEFAULT_SBERT, cache_file: str = EMBED_CACHE_FILE):
        self.model_name = model_name
        self.cache_file = cache_file
        self.embedding_cache: Dict[str, List[float]] = safe_json_read(cache_file) or {}
        self.logger = self._setup_logger()
        self.model = None
        self.embedding_dimension = TFIDF_DIM
        self._load_model()

    def _load_model(self):
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            # flexible id handling
            model_id = self.model_name
            if not model_id.startswith("sentence-transformers/") and "/" not in model_id:
                # try both variants
                try:
                    self.model = SentenceTransformer(f"sentence-transformers/{model_id}")
                except Exception:
                    self.model = SentenceTransformer(model_id)
            else:
                self.model = SentenceTransformer(model_id)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded; dim={self.embedding_dimension}")
        except Exception as e:
            self.logger.warning(f"Failed to load SBERT model: {e}. Using TF-IDF fallback.")
            self.model = None
            self.embedding_dimension = TFIDF_DIM
            self._tfidf = TfidfVectorizer(max_features=TFIDF_DIM)

    def _save_cache(self):
        try:
            safe_json_write(self.cache_file, self.embedding_cache)
        except Exception as e:
            self.logger.debug(f"Could not save cache: {e}")

    def _cache_key(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def encode_skills(self, skills: List[str], use_cache: bool = True, show_progress: bool = False) -> np.ndarray:
        """
        Encode list of skills into embeddings. Uses a JSON cache to save/retrieve vectors.
        Returns numpy array (n_skills, dim)
        """
        if not isinstance(skills, list):
            skills = [skills]
        if len(skills) == 0:
            return np.zeros((0, self.embedding_dimension))

        # prepare result
        results = [None] * len(skills)
        to_compute = []
        to_compute_idx = []

        for i, s in enumerate(skills):
            key = self._cache_key(s)
            if use_cache and key in self.embedding_cache:
                try:
                    arr = np.array(self.embedding_cache[key], dtype=float)
                    results[i] = arr
                except Exception:
                    # invalid cache entry: compute fresh
                    to_compute.append(s)
                    to_compute_idx.append(i)
            else:
                to_compute.append(s)
                to_compute_idx.append(i)

        if to_compute:
            emb_matrix = None
            if self.model is not None:
                try:
                    emb_matrix = self.model.encode(to_compute, convert_to_numpy=True, show_progress_bar=show_progress)
                except Exception as e:
                    self.logger.warning(f"SBERT encode error: {e}; falling back to TF-IDF.")
                    emb_matrix = None

            if emb_matrix is None:
                # TF-IDF fallback
                try:
                    X = self._tfidf.fit_transform(to_compute)
                    arr = X.toarray()
                    # ensure TFIDF_DIM columns
                    if arr.shape[1] < TFIDF_DIM:
                        pad = TFIDF_DIM - arr.shape[1]
                        arr = np.pad(arr, ((0, 0), (0, pad)), 'constant')
                    emb_matrix = arr.astype(float)
                except Exception as e:
                    self.logger.error(f"TF-IDF fallback failed: {e}; generating pseudo-random vectors.")
                    rng = np.random.RandomState(42)
                    emb_matrix = rng.normal(size=(len(to_compute), TFIDF_DIM)).astype(float)

            for local_idx, emb in enumerate(emb_matrix):
                global_idx = to_compute_idx[local_idx]
                results[global_idx] = np.asarray(emb, dtype=float)
                # add to cache
                try:
                    self.embedding_cache[self._cache_key(to_compute[local_idx])] = results[global_idx].tolist()
                except Exception as e:
                    self.logger.debug(f"Cache store failed for {to_compute[local_idx]}: {e}")

            # persist cache
            self._save_cache()

        # stack
        try:
            emb_arr = np.vstack(results)
            return emb_arr
        except Exception as e:
            self.logger.error(f"Could not stack embeddings: {e}")
            # safe fallback
            return np.zeros((len(skills), self.embedding_dimension))

    def get_embedding_for_skill(self, skill: str) -> np.ndarray:
        key = self._cache_key(skill)
        if key in self.embedding_cache:
            try:
                return np.array(self.embedding_cache[key], dtype=float)
            except Exception:
                pass
        emb = self.encode_skills([skill])[0]
        self.embedding_cache[key] = emb.tolist()
        self._save_cache()
        return emb

    def clear_cache(self):
        self.embedding_cache = {}
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            self.logger.info("Embedding cache cleared.")
        except Exception as e:
            self.logger.debug(f"remove cache file failed: {e}")

    def _setup_logger(self) -> logging.Logger:
        logger_local = logging.getLogger("SentenceBERTEncoder")
        if not logger_local.handlers:
            logger_local.setLevel(logging.INFO)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger_local.addHandler(h)
        return logger_local

# ----------------- SkillExtractor -----------------
class SkillExtractor:
    """
    Multi-method skill extractor:
    - If input is skill-list, returns cleaned list
    - Keyword split and normalization
    - Optional spaCy noun-chunk extraction
    - Then deduplicate with fuzzy rules + semantic dedupe using SBERT
    """
    def __init__(self, encoder: SentenceBERTEncoder, lowercase: bool = True):
        self.encoder = encoder
        self.lowercase = lowercase
        self.spacy_nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.spacy_nlp = None

    def extract_from_text(self, text: str, method: str = "keyword") -> List[str]:
        """
        method: 'keyword' | 'spacy' | 'lines'
        """
        if not text or not isinstance(text, str):
            return []
        if method == "lines":
            items = [clean_text(line, lowercase=self.lowercase) for line in text.splitlines() if line.strip()]
            return self._dedupe(items)
        if method == "spacy" and self.spacy_nlp is not None:
            doc = self.spacy_nlp(text)
            candidates = []
            # get noun_chunks and proper nouns and ORG/PRODUCT entities
            for nc in doc.noun_chunks:
                candidates.append(clean_text(nc.text, lowercase=self.lowercase))
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT", "NORP", "TECHNOLOGY", "WORK_OF_ART"}:
                    candidates.append(clean_text(ent.text, lowercase=self.lowercase))
            return self._dedupe(candidates)
        # default keyword split: split by comma, semicolon, slash, newline and parentheses
        items = re.split(r"[,;/\n()\-]+", text)
        items = [clean_text(i, lowercase=self.lowercase) for i in items if i.strip()]
        return self._dedupe(items)

    def _dedupe(self, items: List[str], fuzzy_thresh: float = 0.85, semantic_thresh: float = 0.88) -> List[str]:
        """
        Deduplicate a list of strings via:
         - normalization equality
         - difflib similarity (fuzzy)
         - semantic similarity using SBERT embeddings (for expensive duplicates)
        """
        # normalize
        normalized = []
        for it in items:
            if not it:
                continue
            s = re.sub(r"\s+", " ", it).strip()
            normalized.append(s)

        # quick fuzzy dedupe with SequenceMatcher
        unique = []
        for s in normalized:
            found = False
            for u in unique:
                if similar_ratio(s, u) >= fuzzy_thresh:
                    found = True
                    break
            if not found:
                unique.append(s)

        # semantic dedupe: compute embeddings for unique and compare
        if len(unique) <= 1:
            return unique
        try:
            embs = self.encoder.encode_skills(unique, use_cache=True, show_progress=False)
            keep = []
            for i, s in enumerate(unique):
                duplicate = False
                for k_idx in keep:
                    sim = cosine_similarity([embs[i]], [embs[k_idx]])[0][0]
                    if sim >= semantic_thresh:
                        duplicate = True
                        break
                if not duplicate:
                    keep.append(i)
            final = [unique[i] for i in keep]
            return final
        except Exception as e:
            logger.warning(f"Semantic dedupe failed: {e}")
            return unique

# ----------------- Similarity & Gap Analyzer -----------------
@dataclass
class SkillMatch:
    jd_skill: str
    resume_skill: Optional[str]
    similarity: float
    status: str  # Covered / Missing
    importance: float
    gap_score: float
    evidence: Optional[str] = None  # evidence sentence if available

class SkillGapEngine:
    def __init__(self, encoder: SentenceBERTEncoder, threshold: float = 0.6):
        self.encoder = encoder
        self.threshold = threshold
        self.tfidf = TfidfVectorizer(max_features=TFIDF_DIM)

    def compute(self, resume_skills: List[str], jd_skills: List[str], jd_importance: Optional[List[float]] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
        """
        Returns:
            df: detailed DataFrame (JD Skill, Best Match, Similarity, Status, Importance, gap_score)
            sim_matrix: (len(jd) x len(resume)) matrix or (len(jd), len(resume))
            resume_processed, jd_processed lists
        """
        # preprocess: normalize and remove duplicates via extractor not included here
        resume = [clean_text(s) for s in resume_skills]
        jd = [clean_text(s) for s in jd_skills]
        if len(jd) == 0:
            return pd.DataFrame(), np.zeros((0,0)), resume, jd

        # dedupe near duplicates (diff approach) to reduce matrix size
        # Use simple technique: create unique lists preserving order
        def unique_preserve(seq):
            seen = set()
            out = []
            for s in seq:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        resume_u = unique_preserve(resume)
        jd_u = unique_preserve(jd)

        # embed
        emb_resume = self.encoder.encode_skills(resume_u, use_cache=True, show_progress=False)
        emb_jd = self.encoder.encode_skills(jd_u, use_cache=True, show_progress=False)

        # compute similarity matrix (jd x resume)
        try:
            sim = cosine_similarity(emb_jd, emb_resume)  # shape (len(jd), len(resume))
        except Exception as e:
            logger.error(f"cosine_similarity failed: {e}")
            # fallback by transposing
            sim = np.zeros((len(jd_u), len(resume_u)))

        # compute best matches per JD skill
        rows = []
        importance_list = []
        if jd_importance and len(jd_importance) == len(jd_u):
            importance_list = jd_importance
        else:
            # importance derived from JD frequency (if duplicates present), fallback to 1.0
            freq_map = {k: jd.count(k) for k in set(jd)}
            importance_list = [float(freq_map.get(k, 1)) for k in jd_u]

        # normalize importance to [0,1]
        imp_arr = np.array(importance_list, dtype=float)
        if np.max(imp_arr) > 0:
            imp_arr = imp_arr / np.max(imp_arr)

        for j_idx, j_skill in enumerate(jd_u):
            col = sim[j_idx, :] if sim.size else np.array([])
            if col.size:
                best_i = int(np.argmax(col))
                best_score = float(col[best_i])
                best_resume = resume_u[best_i]
            else:
                best_score = 0.0
                best_resume = None
            importance = float(imp_arr[j_idx]) if len(imp_arr) > j_idx else 1.0
            status = "Covered" if best_score >= self.threshold else "Missing"
            gap_score = round(importance * (1.0 - best_score), 4)
            rows.append({
                "JD Skill": j_skill,
                "Best Resume Match": best_resume,
                "Similarity": round(best_score, 4),
                "Status": status,
                "Importance": round(importance, 4),
                "gap_score": gap_score
            })
        df = pd.DataFrame(rows).sort_values("gap_score", ascending=False).reset_index(drop=True)
        overall_weighted = 0.0
        try:
            # weighted average similarity
            weights = df["Importance"].astype(float).values
            sims = df["Similarity"].astype(float).values
            if np.sum(weights) > 0:
                overall_weighted = float(np.sum(weights * sims) / np.sum(weights)) * 100.0
            else:
                overall_weighted = float(np.mean(sims)) * 100.0
        except Exception as e:
            logger.warning(f"overall weighted calc failed: {e}")
            overall_weighted = float(df["Similarity"].mean() * 100 if not df.empty else 0.0)
        # return df, sim matrix (jd x resume), resume_u, jd_u
        return df, sim, resume_u, jd_u, overall_weighted

# ----------------- LearningPathGenerator (with render_ui) -----------------
class LearningPathGenerator:
    def __init__(self):
        self.logger = logging.getLogger("LearningPathGenerator")
        # small curated resource DB
        self.resources = [
            {"skill": "python", "course": "Python for Everybody", "platform":"Coursera", "difficulty":"Beginner", "weeks":4},
            {"skill": "machine learning", "course": "Machine Learning (Andrew Ng)", "platform":"Coursera", "difficulty":"Advanced", "weeks":12},
            {"skill": "tensorflow", "course": "TensorFlow Developer Certificate", "platform":"TensorFlow", "difficulty":"Intermediate", "weeks":8},
            {"skill": "aws", "course": "AWS Solutions Architect", "platform":"AWS", "difficulty":"Intermediate", "weeks":8},
            {"skill": "docker", "course": "Docker Mastery", "platform":"Udemy", "difficulty":"Beginner", "weeks":3},
            {"skill": "sql", "course": "Advanced SQL", "platform":"DataCamp", "difficulty":"Intermediate", "weeks":4},
            {"skill": "statistics", "course": "Statistics with Python", "platform":"Coursera", "difficulty":"Beginner", "weeks":4},
            {"skill": "deep learning", "course": "Deep Learning Specialization", "platform":"Coursera", "difficulty":"Advanced", "weeks":10},
            {"skill": "data analysis", "course": "Pandas Data Analysis", "platform":"Udemy", "difficulty":"Beginner", "weeks":4}
        ]
        # try to prepare small encoder for semantic matching between missing skill and resources if SBERT available
        try:
            self.encoder = SentenceTransformer(DEFAULT_SBERT)
            self.resource_emb = self.encoder.encode([r["skill"] for r in self.resources], convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            self.encoder = None
            self.resource_emb = None

    def recommend(self, missing_skills: List[str], top_k: int = 3) -> pd.DataFrame:
        recs = []
        if not missing_skills:
            return pd.DataFrame()
        for ms in missing_skills:
            ms_clean = clean_text(ms)
            # semantic matching if embeddings available
            if self.resource_emb is not None and self.encoder is not None:
                try:
                    emb = self.encoder.encode([ms_clean], convert_to_numpy=True, show_progress_bar=False)[0]
                    sims = cosine_similarity([emb], self.resource_emb)[0]
                    top_idxs = np.argsort(sims)[::-1][:top_k]
                    for idx in top_idxs:
                        r = self.resources[idx]
                        recs.append({
                            "missing_skill": ms,
                            "course": r["course"],
                            "platform": r["platform"],
                            "difficulty": r["difficulty"],
                            "weeks": r["weeks"],
                            "score": float(sims[idx])
                        })
                except Exception as e:
                    self.logger.debug(f"Semantic recommend failed for {ms}: {e}")
            else:
                # fallback: substring match or first top_k resources
                found = False
                for r in self.resources:
                    if r["skill"] in ms_clean or ms_clean in r["skill"]:
                        recs.append({
                            "missing_skill": ms,
                            "course": r["course"],
                            "platform": r["platform"],
                            "difficulty": r["difficulty"],
                            "weeks": r["weeks"],
                            "score": 1.0
                        })
                        found = True
                if not found:
                    # add top_k generic suggestions
                    for r in self.resources[:top_k]:
                        recs.append({
                            "missing_skill": ms,
                            "course": r["course"],
                            "platform": r["platform"],
                            "difficulty": r["difficulty"],
                            "weeks": r["weeks"],
                            "score": 0.2
                        })
        df = pd.DataFrame(recs).sort_values("score", ascending=False)
        return df

    def render_ui(self, rec_df: pd.DataFrame):
        """Streamlit UI wrapper to show recommendations (table + visuals + actionable cards)"""
        st.markdown("### 🎓 Personalized Learning Recommendations")
        if rec_df is None or rec_df.empty:
            st.info("No recommendations to show. Generate recommendations first.")
            return

        # show top recommended courses
        with st.expander("Top recommendations (table)"):
            st.dataframe(rec_df, use_container_width=True)

        # group by platform
        platform_counts = rec_df["platform"].value_counts().reset_index()
        platform_counts.columns = ["platform", "count"]
        fig = px.pie(platform_counts, values="count", names="platform", title="Recommended Platforms", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # bar for durations
        fig2 = px.bar(rec_df, x="weeks", y="course", color="difficulty", orientation="h", title="Course Duration & Difficulty", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        # actionable cards for top rows
        st.markdown("#### Quick Actions")
        top = rec_df.head(6)
        for _, r in top.iterrows():
            st.markdown(f"""
            <div style="padding:10px;border-radius:8px;background:#07121a;border:1px solid rgba(0,255,255,0.06);margin-bottom:6px;">
            <strong style="color:#00FFFF">{r['missing_skill']}</strong> — <span style="color:#9ad6ff">{r['course']}</span>
            <div style="font-size:12px;color:#9ac7d6">Platform: {r['platform']} • Difficulty: {r['difficulty']} • Est: {r['weeks']} weeks</div>
            <div style="margin-top:6px"><a href='#' style='color:#00ffff;text-decoration:none'>View course</a> • <a href='#' style='color:#9ad6ff'>Add to plan</a></div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- ExportManager (CSV + PDF) -----------------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import pandas as pd
from datetime import datetime
import os

class ExportManager:
    def __init__(self, df, summary_dict=None, title="SkillGap AI Analyzer Report", logo_path=None):
        """
        Handles exporting of analysis results to CSV and PDF.
        :param df: pandas DataFrame containing skill similarity/gap data
        :param summary_dict: dict containing summary info (e.g., overall match %)
        :param title: report title for PDF header
        :param logo_path: optional path to logo (PNG/JPG)
        """
        self.df = df
        self.summary_dict = summary_dict or {}
        self.title = title
        self.logo_path = logo_path

    # =========================
    # Export CSV
    # =========================
    def export_csv(self):
        """Generate CSV file bytes for download."""
        csv_buffer = BytesIO()
        self.df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return csv_buffer.getvalue()

    # =========================
    # Export PDF
    # =========================
    def export_pdf(self):
        """Generate stylized PDF report."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "titleStyle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#00FFFF"),  # Neon Cyan
            alignment=1,  # Center
            spaceAfter=10,
        )
        subtitle_style = ParagraphStyle(
            "subtitleStyle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=colors.whitesmoke,
            alignment=1,
        )
        normal_style = ParagraphStyle(
            "normalStyle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.white,
        )

        # Dark background
        bg_color = colors.HexColor("#0A0F24")  # Deep Dark Blue

        # Title
        elements.append(Paragraph(self.title, title_style))
        elements.append(Spacer(1, 8))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {now}", subtitle_style))
        elements.append(Spacer(1, 12))

        # Optional logo
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                elements.append(Image(self.logo_path, width=80, height=40, hAlign='RIGHT'))
                elements.append(Spacer(1, 8))
            except Exception as e:
                print(f"⚠️ Logo load failed: {e}")

        # Summary
        if self.summary_dict:
            summary_data = [["Metric", "Value"]] + [
                [k, str(v)] for k, v in self.summary_dict.items()
            ]
            summary_table = Table(summary_data, hAlign="LEFT")
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D6EFD")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#111A3A")),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#00FFFF")),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 12))

        # Data Table
        df_data = [list(self.df.columns)] + self.df.values.tolist()
        table = Table(df_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#00FFFF")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#0A0F24")),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#00FFFF")),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Footer
        elements.append(Paragraph("© SkillGap AI Analyzer | Generated Report", subtitle_style))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_bg, onLaterPages=self._add_bg)

        pdf_value = buffer.getvalue()
        buffer.close()
        return pdf_value

    # =========================
    # Background Drawer
    # =========================
    def _add_bg(self, canvas, doc):
        """Draw dark neon background for each PDF page."""
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#0A0F24"))
        canvas.rect(0, 0, A4[0], A4[1], fill=True, stroke=False)
        canvas.restoreState()




# ----------------- Visualization Helpers -----------------
class Visuals:
    def __init__(self):
        sns.set_style("darkgrid")

    def show_summary_card(self, overall_pct: float, df: pd.DataFrame):
        covered = df[df["Status"] == "Covered"].shape[0]
        missing = df[df["Status"] == "Missing"].shape[0]
        st.markdown(f"""
        <div style="padding:12px;border-radius:10px;background:#06121a;border:1px solid rgba(0,255,255,0.06);">
          <div style="font-size:14px;color:#66fff1;font-weight:700">Overall Match</div>
          <div style="font-size:28px;color:#00ffff;font-weight:800">{overall_pct:.2f}%</div>
          <div style="font-size:12px;color:#9ac7d6">Covered: {covered} • Missing: {missing}</div>
        </div>
        """, unsafe_allow_html=True)

    def plot_heatmap(self, resume_list: List[str], jd_list: List[str], sim_matrix: np.ndarray):
        if sim_matrix is None or sim_matrix.size == 0:
            st.info("Similarity matrix not available")
            return
        # sim_matrix is shape (len(jd), len(resume)) per engine
        # we'll display a heatmap with JD rows and Resume columns
        # limit sizes for display
        maxdim = 40
        rlist = resume_list[:maxdim]
        jlist = jd_list[:maxdim]
        mat = sim_matrix[:len(jlist), :len(rlist)]
        fig = go.Figure(data=go.Heatmap(
            z=mat,
            x=rlist,
            y=jlist,
            colorscale='Viridis',
            zmin=0, zmax=1,
            text=np.round(mat, 3),
            hovertemplate="JD: %{y}<br>Resume: %{x}<br>Similarity: %{z}"
        ))
        fig.update_layout(title="Similarity Matrix (JD vs Resume)", template="plotly_dark", height=700)
        st.plotly_chart(fig, use_container_width=True)

    def plot_radar(self, df: pd.DataFrame):
        # Radar across categories - but here we don't have explicit category mapping;
        # We'll show top categories by skill groups (approx) by splitting by words (heuristic).
        if df is None or df.empty:
            return
        # take top 6 JD skills by importance or similarity for radial display
        top = df.sort_values("Importance", ascending=False).head(6)
        categories = top["JD Skill"].tolist()
        scores = top["Similarity"].tolist()
        if not categories:
            return
        categories += [categories[0]]
        scores += [scores[0]]
        fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', line_color='#00ffff'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), template="plotly_dark", title="Radar: Top JD Skills vs Similarity")
        st.plotly_chart(fig, use_container_width=True)

    def plot_missing_bar(self, df: pd.DataFrame, top_n: int = 10):
        missing_df = df[df["Status"] == "Missing"].copy()
        if missing_df.empty:
            st.info("No missing skills to show.")
            return
        missing_df["priority_metric"] = missing_df["Importance"] * (1 - missing_df["Similarity"])
        top = missing_df.sort_values("priority_metric", ascending=False).head(top_n)
        fig = px.bar(top, x="priority_metric", y="JD Skill", orientation="h", color="priority_metric", color_continuous_scale="RdYlGn_r", template="plotly_dark", title=f"Top {top_n} Missing Skills by Priority")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    def plot_importance_similarity(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        fig = px.scatter(df, x="Importance", y="Similarity", hover_name="JD Skill", color="Status", size="gap_score", template="plotly_dark", title="Importance vs Similarity")
        st.plotly_chart(fig, use_container_width=True)

# ----------------- App Orchestration (Streamlit) -----------------
class SkillGapApp:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="SkillGap AI Analyzer")
        self.encoder = SentenceBERTEncoder(DEFAULT_SBERT)
        self.extractor = SkillExtractor(self.encoder)
        self.engine = SkillGapEngine(self.encoder, threshold=0.6)
        self.lpgen = LearningPathGenerator()
        self.exporter = None
        self.visuals = Visuals()
        # session defaults
        st.session_state.setdefault("resume_text", "")
        st.session_state.setdefault("jd_text", "")
        st.session_state.setdefault("resume_skills", [])
        st.session_state.setdefault("jd_skills", [])
        st.session_state.setdefault("analysis_df", pd.DataFrame())
        st.session_state.setdefault("sim_matrix", None)
        st.session_state.setdefault("overall_match", 0.0)
        # inject CSS for dark neon
        self._inject_css()

    def _inject_css(self):
        css = """
        <style>
        body { background-color: #0b0f14; color: #cfefff; }
        .stButton>button { background: linear-gradient(90deg,#00e5ff,#0066ff); color:#001; font-weight:700; border-radius:8px; padding:8px 12px; }
        .neon-card { background:#07121a; border-radius:10px; padding:10px; border:1px solid rgba(0,255,255,0.06); }
        .muted { color:#9ac7d6; font-size:12px; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    def sidebar(self):
        st.sidebar.title("SkillGap Controls")
        st.sidebar.markdown("**Model & Preprocessing**")
        model_choice = st.sidebar.selectbox("Embedding model", [DEFAULT_SBERT, "paraphrase-MiniLM-L6-v2"], index=0)
        # If model changed, reload encoder
        if model_choice != self.encoder.model_name:
            # replace encoder with new model choice
            try:
                self.encoder = SentenceBERTEncoder(model_choice)
                self.extractor = SkillExtractor(self.encoder)  # rebind extractor
                self.engine = SkillGapEngine(self.encoder, threshold=self.engine.threshold)
                st.sidebar.success(f"Model switched to {model_choice}")
            except Exception as e:
                st.sidebar.warning(f"Could not switch model: {e}")

        lowercase = st.sidebar.checkbox("Normalize (lowercase)", value=True, key="lowercase")
        use_spacy = st.sidebar.checkbox("Use spaCy extraction (if available)", value=False)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Matching**")
        thresh = st.sidebar.slider("Similarity threshold", 0.1, 0.95, self.engine.threshold, 0.01)
        self.engine.threshold = thresh
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Cache & Export**")
        use_cache = st.sidebar.checkbox("Use embedding cache (file)", True)
        if st.sidebar.button("Clear embedding cache"):
            self.encoder.clear_cache()
            st.experimental_rerun()

    def ingest_panel(self):
        st.header("1 — Upload & Text Ingestion")
        st.markdown("<div class='muted'>Upload resume(s) and job description(s). Accepts CSV (skill lists), TXT or paste text. PDF OCR placeholder available.</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            uploaded_res = st.file_uploader("Upload Resume file (CSV/TXT/PDF) or upload multiple (separate uploads)", type=["csv","txt","pdf"], key="res_file")
            text_area_res = st.text_area("Or paste resume text (optional)", height=180, key="res_area")
        with col2:
            uploaded_jd = st.file_uploader("Upload JD file (CSV/TXT/PDF)", type=["csv","txt","pdf"], key="jd_file")
            text_area_jd = st.text_area("Or paste JD text (optional)", height=180, key="jd_area")

        # parse uploader when user clicks parse
        if st.button("Parse uploads and extract skill lines"):
            resume_skills = []
            jd_skills = []
            # resume
            try:
                if uploaded_res:
                    fname = uploaded_res.name.lower()
                    raw = uploaded_res.read()
                    if fname.endswith(".csv"):
                        try:
                            df = pd.read_csv(io.BytesIO(raw))
                            # attempt guess column
                            col = None
                            for c in df.columns:
                                if "skill" in str(c).lower():
                                    col = c
                                    break
                            if col is None:
                                col = df.columns[0]
                            resume_skills = df[col].astype(str).tolist()
                        except Exception as e:
                            st.warning(f"Resume CSV parse failed: {e}")
                    elif fname.endswith(".pdf"):
                        # placeholder for OCR
                        txt = ""  # ocr_placeholder(raw)
                        # naive split
                        resume_skills = [line.strip() for line in txt.splitlines() if line.strip()]
                    else:
                        try:
                            txt = raw.decode("utf-8")
                        except Exception:
                            txt = raw.decode("latin-1")
                        resume_skills = [line.strip() for line in txt.splitlines() if line.strip()]
                # fallback to text area
                if not resume_skills and text_area_res:
                    resume_skills = [line.strip() for line in text_area_res.splitlines() if line.strip()]
            except Exception as e:
                st.error(f"Resume ingestion failed: {e}")
                resume_skills = []

            # jd
            try:
                if uploaded_jd:
                    fname = uploaded_jd.name.lower()
                    raw = uploaded_jd.read()
                    if fname.endswith(".csv"):
                        try:
                            df = pd.read_csv(io.BytesIO(raw))
                            col = None
                            for c in df.columns:
                                if "skill" in str(c).lower():
                                    col = c
                                    break
                            if col is None:
                                col = df.columns[0]
                            jd_skills = df[col].astype(str).tolist()
                        except Exception as e:
                            st.warning(f"JD CSV parse failed: {e}")
                    elif fname.endswith(".pdf"):
                        txt = ""  # ocr_placeholder
                        jd_skills = [line.strip() for line in txt.splitlines() if line.strip()]
                    else:
                        try:
                            txt = raw.decode("utf-8")
                        except Exception:
                            txt = raw.decode("latin-1")
                        jd_skills = [line.strip() for line in txt.splitlines() if line.strip()]
                if not jd_skills and text_area_jd:
                    jd_skills = [line.strip() for line in text_area_jd.splitlines() if line.strip()]
            except Exception as e:
                st.error(f"JD ingestion failed: {e}")
                jd_skills = []

            # store
            st.session_state["resume_skills"] = resume_skills
            st.session_state["jd_skills"] = jd_skills
            st.success(f"Parsed: {len(resume_skills)} resume lines, {len(jd_skills)} JD lines")

        # show preview
        st.markdown("### Preview (first 20)")
        c1, c2 = st.columns(2)
        with c1:
            st.write(st.session_state.get("resume_skills", [])[:20])
        with c2:
            st.write(st.session_state.get("jd_skills", [])[:20])

    def extraction_panel(self):
        st.header("2 — Skill Extraction")
        st.markdown("<div class='muted'>Choose extraction method. Use spaCy (if installed) for noun-chunk/NER extraction, or keyword fallback. Then deduplicate semantically.</div>", unsafe_allow_html=True)
        method = st.selectbox("Extraction method", ["Use provided lines", "Keyword split", "spaCy (if available)"])
        normalize_flag = st.checkbox("Normalize (lowercasing and punctuation removal)", value=True)

        if st.button("Run extraction"):
            resume_raw = list_to_text(st.session_state.get("resume_skills", []))
            jd_raw = list_to_text(st.session_state.get("jd_skills", []))
            if method == "Use provided lines":
                extracted_resume = [clean_text(s, lowercase=normalize_flag) for s in st.session_state.get("resume_skills", []) if s.strip()]
                extracted_jd = [clean_text(s, lowercase=normalize_flag) for s in st.session_state.get("jd_skills", []) if s.strip()]
            elif method == "Keyword split":
                def split_clean(txt):
                    items = re.split(r"[,;/\n()\-]+", txt)
                    return [clean_text(i, lowercase=normalize_flag) for i in items if i.strip()]
                extracted_resume = split_clean(resume_raw)
                extracted_jd = split_clean(jd_raw)
            else:
                # spaCy extraction if available else fallback
                extracted_resume = self.extractor.extract_from_text(resume_raw, method="spacy" if _SPACY_AVAILABLE else "keyword")
                extracted_jd = self.extractor.extract_from_text(jd_raw, method="spacy" if _SPACY_AVAILABLE else "keyword")

            # dedupe using extractor's semantic dedupe
            extracted_resume = self.extractor._dedupe(extracted_resume)
            extracted_jd = self.extractor._dedupe(extracted_jd)
            st.session_state["resume_skills_processed"] = extracted_resume
            st.session_state["jd_skills_processed"] = extracted_jd
            st.success(f"Extraction complete — {len(extracted_resume)} resume skills, {len(extracted_jd)} JD skills")

        st.markdown("### Extracted Preview")
        st.dataframe({
            "Resume (extracted)": st.session_state.get("resume_skills_processed", [])[:50],
            "JD (extracted)": st.session_state.get("jd_skills_processed", [])[:50]
        }, use_container_width=True)

    def analyze_panel(self):
        st.header("3 — Skill Gap Analysis & Matching")
        st.markdown("<div class='muted'>Compute S-BERT embeddings, cosine similarity, weighted overall match, and ranked gaps.</div>", unsafe_allow_html=True)
        if st.button("Compute similarity & rank gaps"):
            resume_skills = st.session_state.get("resume_skills_processed", []) or st.session_state.get("resume_skills", [])
            jd_skills = st.session_state.get("jd_skills_processed", []) or st.session_state.get("jd_skills", [])
            if not resume_skills or not jd_skills:
                st.error("Please provide/extract resume and JD skills first.")
                return
            try:
                # optional importance input
                imp_text = st.text_input("Optional importance weights (comma-separated aligned to JD skills)", "")
                imp_vals = None
                if imp_text.strip():
                    try:
                        tokens = [float(x.strip()) for x in imp_text.split(",") if x.strip()]
                        if len(tokens) == len(jd_skills):
                            imp_vals = tokens
                        else:
                            st.warning("Importance length mismatch — ignoring.")
                    except Exception:
                        st.warning("Invalid importance values; ignoring.")

                df, sim_matrix, resume_u, jd_u, overall_weighted = self.engine.compute(resume_skills, jd_skills, jd_importance=imp_vals)
                # store
                st.session_state["analysis_df"] = df
                st.session_state["sim_matrix"] = sim_matrix
                st.session_state["resume_u"] = resume_u
                st.session_state["jd_u"] = jd_u
                st.session_state["overall_match"] = overall_weighted
                st.success(f"Matching complete — Overall weighted match: {overall_weighted:.2f}%")
            except Exception as e:
                logger.exception("Analysis failed")
                st.error(f"Analysis failed: {e}")

        # show summary & top 3 high priority missing
        df = st.session_state.get("analysis_df", pd.DataFrame())
        overall = st.session_state.get("overall_match", 0.0)
        if not df.empty:
            self.visuals.show_summary_card(overall, df)
            # Top 3 missing high-importance skills (priority metric = importance*(1-similarity))
            df_missing = df[df["Status"] == "Missing"].copy()
            if not df_missing.empty:
                df_missing["priority_metric"] = df_missing["Importance"] * (1 - df_missing["Similarity"])
                top3 = df_missing.sort_values("priority_metric", ascending=False).head(3)
                st.markdown("### Top 3 High-Importance Missing Skills")
                for i, r in top3.iterrows():
                    st.markdown(f"- **{r['JD Skill']}** — importance: {r['Importance']:.2f} • similarity: {r['Similarity']:.2f} • gap_score: {r['gap_score']:.4f}")

            st.markdown("### Detailed table (JD skill → best resume match)")
            st.dataframe(df, use_container_width=True)

    def visualize_panel(self):
        st.header("4 — Visualizations")
        df = st.session_state.get("analysis_df", pd.DataFrame())
        sim = st.session_state.get("sim_matrix", None)
        resume = st.session_state.get("resume_u", [])
        jd = st.session_state.get("jd_u", [])
        if df is None or df.empty:
            st.info("Run analysis first.")
            return

        # layout
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Radar (Top JD)")
            self.visuals.plot_radar(df)
            st.subheader("Importance vs Similarity")
            self.visuals.plot_importance_similarity(df)
        with col2:
            st.subheader("Similarity Heatmap")
            self.visuals.plot_heatmap(resume, jd, sim)
            st.subheader("Top Missing Skills")
            self.visuals.plot_missing_bar(df, top_n=12)

    def actions_panel(self):
        st.header("5 — Actions, Recommendations & Export")
        df = st.session_state.get("analysis_df", pd.DataFrame())
        overall = st.session_state.get("overall_match", 0.0)
        if df is None or df.empty:
            st.info("Run analysis first.")
            return

        # High priority gaps with suggested actions
        st.subheader("High priority gaps")
        df_missing = df[df["Status"] == "Missing"].copy()
        if df_missing.empty:
            st.success("No missing skills detected — good match!")
        else:
            df_missing["priority_metric"] = df_missing["Importance"] * (1 - df_missing["Similarity"])
            top = df_missing.sort_values("priority_metric", ascending=False)
            # suggested actions naive mapping
            def suggestion(skill):
                skill_l = skill.lower()
                if any(k in skill_l for k in ["python","sql","java","c++","c#"]):
                    return "Learn via project + certification; add code samples to resume"
                if any(k in skill_l for k in ["aws","azure","gcp","cloud"]):
                    return "Take cloud certification (cloud practitioner/associate) and add hands-on labs"
                if any(k in skill_l for k in ["docker","kubernetes","container"]):
                    return "Complete Docker/Kubernetes labs; include deployment project"
                if any(k in skill_l for k in ["ml","machine learning","deep learning","tensorflow","pytorch"]):
                    return "Take ML specialization course + small project (notebook) demonstrating model"
                return "Take an online course and build a small project; add to resume"

            for i, r in top.head(8).iterrows():
                st.markdown(f"**{r['JD Skill']}** — importance {r['Importance']:.2f}, similarity {r['Similarity']:.2f}")
                st.caption(f"Suggested action: {suggestion(r['JD Skill'])}")

        # Learning path recommendations
        missing_list = df_missing["JD Skill"].tolist()
        if st.button("Generate Learning Recommendations"):
            recs = self.lpgen.recommend(missing_list, top_k=3)
            self.lpgen.render_ui(recs)

        # Export options
        exporter = ExportManager(df,{"Overall Match": f"{overall:.2f}%"},title="SkillGap AI Analyzer — Report")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
            "⬇️ Download CSV",
            data=exporter.export_csv(),
            file_name=f"skillgap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True)
        with col2:
            st.download_button(
            "📘 Download PDF",
            data=exporter.export_pdf(),
            file_name=f"skillgap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True)



    def diagnostics_panel(self):
        st.header("Diagnostics & Logs")
        st.markdown("Engine settings and recent logs.")
        st.write({
            "SBERT model": self.encoder.model_name,
            "Embedding dim": self.encoder.embedding_dimension,
            "Cache entries": len(self.encoder.embedding_cache)
        })
        if st.button("Show recent logs"):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    text = f.read()[-20000:]
                st.text_area("Recent logs", text, height=400)
            except Exception as e:
                st.warning(f"Cannot read logs: {e}")

    def run(self):
        st.markdown("<div style='padding:10px;border-radius:10px;background:linear-gradient(90deg,#001220,#001a24);'><h2 style='color:#00ffff'>SkillGap AI Analyzer</h2><div style='color:#9ad6ff'>Dark Neon — S-BERT matching & gap analysis</div></div>", unsafe_allow_html=True)
        self.sidebar()
        pages = [
            "1 — Ingest & Preview",
            "2 — Extract Skills",
            "3 — Analyze & Rank",
            "4 — Visualize",
            "5 — Actions & Export",
            "6 — Diagnostics"
        ]
        page = st.sidebar.radio("Navigate", pages, index=0)
        if page == pages[0]:
            self.ingest_panel()
        elif page == pages[1]:
            self.extraction_panel()
        elif page == pages[2]:
            self.analyze_panel()
        elif page == pages[3]:
            self.visualize_panel()
        elif page == pages[4]:
            self.actions_panel()
        else:
            self.diagnostics_panel()

# Helper for text area combining
def list_to_text(lst: List[str]) -> str:
    return "\n".join(lst)

# ----------------- Main -----------------
def main():
    app = SkillGapApp()
    app.run()

if __name__ == "__main__":
    main()

