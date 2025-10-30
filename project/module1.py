"""
Complete Document Processing Pipeline for AI Skill Gap Analyzer
Milestone 1: Data Ingestion, Parsing, Cleaning, Metrics & Export

Run with:
streamlit run complete_pipeline.py
"""

import streamlit as st
import PyPDF2
import docx
import pandas as pd
import re 
import json
from io import BytesIO
from datetime import datetime

# ------------------------
# Helper Classes
# ------------------------

class DocumentUploader:
    """Handles file upload and validation"""
    def __init__(self):
        self.supported_formats = ['pdf', 'docx', 'txt']
        self.max_file_size = 10 * 1024 * 1024  # 10 MB

    def upload_files(self):
        st.header("🚀Step 1: Upload Documents📄")
        st.markdown(
            "Upload resumes and job descriptions (PDF, DOCX, TXT). Max 10MB each.\n"
        )

        # Resume Upload Section
        st.subheader("📂 Upload Resumes")
        resume_files = st.file_uploader(
            "Select one or more resume files",
            type=self.supported_formats,
            accept_multiple_files=True,
            key="resumes"
        )
        st.markdown("---")

        # Job Description Upload Section
        st.subheader("📂 Upload Job Descriptions")
        job_files = st.file_uploader(
            "Select one or more job description files",
            type=self.supported_formats,
            accept_multiple_files=True,
            key="jobs"
        )
        st.markdown("---")

        # Process uploaded files
        all_files = []
        if resume_files:
            all_files.extend(self._process_uploaded_files(resume_files, "resume"))
        if job_files:
            all_files.extend(self._process_uploaded_files(job_files, "job_description"))
        return all_files

    def _process_uploaded_files(self, files, doc_type):
        processed_files = []
        for file in files:
            if file.size > self.max_file_size:
                st.error(f"{file.name}: Exceeds 10MB limit")
                continue
            ext = file.name.split('.')[-1].lower()
            if ext not in self.supported_formats:
                st.error(f"{file.name}: Unsupported format")
                continue
            content = file.read()
            processed_files.append({
                'name': file.name,
                'type': doc_type,
                'content': content,
                'format': ext,
                'upload_time': datetime.now()
            })
        return processed_files


class TextExtractor:
    """Extract text from PDF, DOCX, TXT"""
    def extract(self, file_info):
        fmt = file_info['format']
        try:
            if fmt == 'pdf':
                return self._from_pdf(file_info['content'])
            elif fmt == 'docx':
                return self._from_docx(file_info['content'])
            elif fmt == 'txt':
                return self._from_txt(file_info['content'])
            else:
                return ""
        except Exception as e:
            st.warning(f"Extraction failed for {file_info['name']}: {e}")
            return ""

    def _from_pdf(self, content):
        text = ""
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}"
        return text.strip()

    def _from_docx(self, content):
        doc = docx.Document(BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    def _from_txt(self, content):
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return content.decode(enc)
            except:
                continue
        return content.decode('utf-8', errors='replace')


class TextCleaner:
    """Clean and preprocess text"""

    def clean(self, text, doc_type='general'):
        if not text.strip():
            return ""

        # Normalize line breaks and remove extra white spaces
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)

        # Remove non-ASCII characters & weird encodings (e.g., â€™, Â)
        text = text.encode('ascii', errors='ignore').decode()

        # Remove unwanted symbols, headers, footers, artifacts
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # ASCII control chars
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
        text = re.sub(r'\b(?:Confidential|Resume|Curriculum Vitae|CV)\b', '', text, flags=re.IGNORECASE)

        # Remove bullet points, extra punctuation clutter
        text = re.sub(r'[•●★◆]', '', text)
        text = re.sub(r'[^\w\s,.!?%+-]', '', text)  # remove unusual symbols but keep basic ones

        # Clean up spacing again
        text = re.sub(r'\s{2,}', ' ', text)

        return text.strip()

    def extract_skills(self, text):
        # Expanded skillset for better coverage
        skills_list = [
            'Python', 'C++', 'C#', 'Java', 'JavaScript', 'HTML', 'CSS', 'SQL',
            'MySQL', 'PostgreSQL', 'Excel', 'Pandas', 'NumPy',
            'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Visualization',
            'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch',
            'Power BI', 'Tableau', 'AWS', 'Azure', 'Git', 'Docker', 'Kubernetes',
            'NLP', 'Computer Vision', 'Flask', 'Django'
        ]

        found = [skill for skill in skills_list if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
        return list(set(found))  # Remove duplicates



# ------------------------
# Pipeline Functions
# ------------------------

def process_documents(files):
    extractor = TextExtractor()
    cleaner = TextCleaner()
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_files = len(files)
    for i, file in enumerate(files):
        status_text.text(f"Processing {file['name']} ({i+1}/{total_files})")
        raw_text = extractor.extract(file)
        cleaned_text = cleaner.clean(raw_text, file['type'])
        skills = cleaner.extract_skills(cleaned_text)
        results.append({
            'file_name': file['name'],
            'document_type': file['type'],
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'word_count_before': len(raw_text.split()),
            'char_count_before': len(raw_text),
            'word_count_after': len(cleaned_text.split()),
            'char_count_after': len(cleaned_text),
            'skills': skills,
            'success': True
        })
        progress_bar.progress((i+1)/total_files)
    progress_bar.empty()
    status_text.empty()
    return results


# ------------------------
# Streamlit UI
# ------------------------

def main():
    st.set_page_config(page_title="AI Skill Gap Analyzer", layout="wide")

    # ------------------------
    # Add page background and custom styles
    # ------------------------
    page_bg_color = "#F5F5F5"  # light grey
    sidebar_bg_color = "#FFFFFF"  # white
    st.markdown(
        f"""
        <style>
        /* Page background */
        .stApp {{
            background-color: {page_bg_color};
        }}
        /* Sidebar style */
        .css-1d391kg .css-1d391kg {{
            background-color: {sidebar_bg_color};
        }}
        /* Headings style */
        h1, h2, h3, h4 {{
            color: #333333;
        }}
        /* Upload sections cards */
        .upload-card {{
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            margin-bottom: 25px;
        }}
        /* Step highlight in sidebar */
        .step-card {{
            padding: 10px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        </style>
        """, unsafe_allow_html=True
    )

    # ------------------------
    # Initialize session state
    # ------------------------
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []

    # ------------------------
    # Sidebar Navigation
    # ------------------------
    with st.sidebar:
        st.title("📌 AI Skill Gap Analyzer")
        steps = ["Upload", "Processing", "Metrics", "Export"]
        step_icons = ["📄", "⚡", "📊", "💾"]
        step_colors = ["#FF6F61", "#6BCB77", "#4D96FF", "#FFD93D"]

        st.markdown("### 🔹 Pipeline Steps")
        for i, step_name in enumerate(steps, start=1):
            color = step_colors[i-1]
            icon = step_icons[i-1]
            if i == st.session_state.current_step:
                st.markdown(
                    f"<div class='step-card' style='background-color:{color};'>{icon} Step {i}: {step_name}</div>", 
                    unsafe_allow_html=True
                )
            else:
                if st.button(f"{icon} Step {i}: {step_name}", key=f"step_{i}"):
                    st.session_state.current_step = i

        st.markdown("---")
        uploaded_count = len(st.session_state.uploaded_files)
        processed_count = len(st.session_state.processed_docs)
        st.markdown(f"- **Uploaded Files:** {uploaded_count}")
        st.markdown(f"- **Processed Docs:** {processed_count}")

    # ----------------------
    # Step 1: Upload
    # ----------------------
    if st.session_state.current_step == 1:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        uploader = DocumentUploader()
        st.session_state.uploaded_files = uploader.upload_files()
        st.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.uploaded_files:
            if st.button("➡️ Next"):
                st.session_state.current_step = 2

    # ----------------------
    # Step 2: Processing
    # ----------------------
    elif st.session_state.current_step == 2:
        st.title("⚡ Step 2: Processing")
        if not st.session_state.uploaded_files:
            st.warning("No files uploaded. Go back to Step 1.")
        else:
            if st.button("Process Documents"):
                st.session_state.processed_docs = process_documents(st.session_state.uploaded_files)
                st.success("Processing complete!")
            if st.session_state.processed_docs:
                if st.button("➡️ Next"):
                    st.session_state.current_step = 3
        if st.button("⬅️ Back"):
            st.session_state.current_step = 1

    # ----------------------
    # Step 3: Metrics
    # ----------------------
    elif st.session_state.current_step == 3:
        st.title("📊 Step 3: Metrics")
        docs = st.session_state.processed_docs
        if not docs:
            st.warning("No processed documents available. Go back to Step 2.")
        else:
            for doc in docs:
                st.markdown('<div class="upload-card">', unsafe_allow_html=True)
                st.markdown(f"### 📄 {doc['file_name']} ({doc['document_type']})")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Words Before Cleaning: {doc['word_count_before']}")
                    st.info(f"Chars Before Cleaning: {doc['char_count_before']}")
                with col2:
                    st.success(f"Words After Cleaning: {doc['word_count_after']}")
                    st.success(f"Chars After Cleaning: {doc['char_count_after']}")
                with col3:
                    st.warning(f"Extracted Skills: {', '.join(doc['skills']) if doc['skills'] else 'None'}")
                st.text_area("Preview Cleaned Text", doc['cleaned_text'][:500]+"..." if len(doc['cleaned_text'])>500 else doc['cleaned_text'], height=200)
                st.markdown('</div>', unsafe_allow_html=True)

            col_back, col_next = st.columns([1,1])
            with col_back:
                if st.button("⬅️ Back"):
                    st.session_state.current_step = 2
            with col_next:
                if st.button("➡️ Next"):
                    st.session_state.current_step = 4

    # ----------------------
    # Step 4: Export
    # ----------------------
    elif st.session_state.current_step == 4:
        st.title("💾 Step 4: Export")
        docs = st.session_state.processed_docs
        if not docs:
            st.warning("No processed documents to export.")
        else:
            df_summary = pd.DataFrame([{
                'File Name': d['file_name'],
                'Doc Type': d['document_type'],
                'Words Before': d['word_count_before'],
                'Chars Before': d['char_count_before'],
                'Words After': d['word_count_after'],
                'Chars After': d['char_count_after'],
                'Skills': ", ".join(d['skills'])
            } for d in docs])
            st.subheader("Processed Documents Summary")
            st.dataframe(df_summary)

            col1, col2,col3 = st.columns(3)
            with col1:
                csv_data = pd.DataFrame([{
                    'filename': d['file_name'],
                    'document_type': d['document_type'],
                    'raw_text': d['raw_text'],
                    'cleaned_text': d['cleaned_text'],
                    'word_count_before': d['word_count_before'],
                    'char_count_before': d['char_count_before'],
                    'word_count_after': d['word_count_after'],
                    'char_count_after': d['char_count_after'],
                    'skills': ", ".join(d['skills'])
                } for d in docs]).to_csv(index=False)
                st.download_button("📥 Download CSV", csv_data, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with col2:
                json_data = json.dumps([{
                    'filename': d['file_name'],
                    'document_type': d['document_type'],
                    'raw_text': d['raw_text'],
                    'cleaned_text': d['cleaned_text'],
                    'word_count_before': d['word_count_before'],
                    'char_count_before': d['char_count_before'],
                    'word_count_after': d['word_count_after'],
                    'char_count_after': d['char_count_after'],
                    'skills': d['skills']
                } for d in docs], indent=2)
                st.download_button("📥 Download JSON", json_data, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with col3:
                try:
                    excel_buffer = BytesIO()
                    pd.DataFrame([{
                        'filename': d['file_name'],
                        'document_type': d['document_type'],
                        'raw_text': d['raw_text'],
                        'cleaned_text': d['cleaned_text'],
                        'word_count_before': d['word_count_before'],
                        'char_count_before': d['char_count_before'],
                        'word_count_after': d['word_count_after'],
                        'char_count_after': d['char_count_after'],
                        'skills': ", ".join(d['skills'])
                    } for d in docs]).to_csv(excel_buffer, index=False)
                    st.download_button("📥 Download Excel", excel_buffer, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                except Exception as e:
                    st.error(f"Excel download failed: {e}")
            if st.button("⬅️ Back"):
                st.session_state.current_step = 3

if __name__ == "__main__":
    main()
