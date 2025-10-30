#main

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from skill_extractor import SkillExtractor
from bert_embedder import SentenceBERTEmbedder
from ner_trainer import CustomNERTrainer
from annotation_ui import AnnotationInterface
from visualizer import SkillVisualizer
from exporter import Exporter


@st.cache_resource
def init_modules():
    return {
        "extractor": SkillExtractor(),
        "embedder": SentenceBERTEmbedder(),
        "ner_trainer": CustomNERTrainer(),
        "annotator": AnnotationInterface(),
        "visualizer": SkillVisualizer()
    }


modules = init_modules()

steps = [
    "Input Data",
    "Extract Skills",
    "Embeddings",
    "Train Custom NER",
    "Annotation",
    "Visualization",
    "Export Data"
]

if "step_index" not in st.session_state:
    st.session_state.step_index = 0


def go_next():
    if st.session_state.step_index < len(steps) - 1:
        st.session_state.step_index += 1


def go_prev():
    if st.session_state.step_index > 0:
        st.session_state.step_index -= 1


def input_step():
    st.header("Step 1: Input Data")
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"], horizontal=True)
    text_input = ""
    doc_type = "resume"
    if input_method == "Paste Text":
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area("Paste resume or job description here:", height=200)
        with col2:
            doc_type = st.selectbox("Document Type:", ["resume", "job_description"])
    else:
        uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
        doc_type = st.selectbox("Document Type:", ["resume", "job_description"])
        if uploaded_file:
            text_input = uploaded_file.getvalue().decode("utf-8")
    return text_input, doc_type


def extraction_step(text_input, doc_type):
    st.header("Step 2: Extract Skills")
    run_extract = st.button("Extract Skills")
    if run_extract:
        if not text_input.strip():
            st.warning("Please provide some text input before extraction.")
            return None
        with st.spinner("Extracting skills..."):
            result = modules["extractor"].extract_skills(text_input, doc_type)
            if result.get("success"):
                st.success(f"Extraction successful! Found {result['statistics']['total_skills']} skills.")
                st.session_state.extraction_results = result
                return result
            else:
                st.error(f"Extraction failed: {result.get('error')}")
                return None
    else:
        return st.session_state.get("extraction_results", None)


def show_summary(result):
    st.subheader("Extraction Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Skills", result['statistics']['total_skills'])
    c2.metric("Technical Skills", result['statistics']['technical_skills'])
    c3.metric("Soft Skills", result['statistics']['soft_skills'])
    avg_conf = sum(result['skill_confidence'].values()) / len(result['skill_confidence']) if result['skill_confidence'] else 0
    c4.metric("Average Confidence", f"{avg_conf:.0%}")


def show_categorized_skills(result):
    st.subheader("Categorized Skills")
    categorized = result['categorized_skills']
    items = list(categorized.items())

    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(items):
                category, skills = items[i + j]
                col.markdown(f"**{category.replace('_', ' ').title()} ({len(skills)})**")
                skill_tags = ""
                for skill in skills[:10]:
                    conf = result['skill_confidence'].get(skill, 0)
                    color_cls = "tech-skill" if category != "soft_skills" else "soft-skill"
                    skill_tags += f'<span class="skill-tag {color_cls}" title="Confidence: {conf:.0%}">{skill}</span> '
                if len(skills) > 10:
                    skill_tags += f'<span class="skill-tag">+{len(skills) - 10} more</span>'
                col.markdown(skill_tags, unsafe_allow_html=True)


def embedding_step():
    st.header("Step 3: Sentence-BERT Embeddings")
    if 'extraction_results' not in st.session_state:
        st.info("Please complete skill extraction first.")
        return
    skills = st.session_state.extraction_results.get('all_skills', [])

    if st.button("Generate Embeddings"):
        with st.spinner("Generating embeddings..."):
            embeddings = modules["embedder"].encode_skills(skills)
            st.session_state.skill_embeddings = embeddings
        st.success(f"Generated embeddings for {len(skills)} skills.")

    if "skill_embeddings" in st.session_state:
        st.subheader("Skill Similarity Calculator")
        col1, col2 = st.columns(2)
        skill1 = col1.selectbox("First skill", skills, key="sim_skill1")
        skill2 = col2.selectbox("Second skill", skills, key="sim_skill2")
        if st.button("Calculate Similarity"):
            sim = modules["embedder"].compute_similarity(skill1, skill2)
            st.metric("Similarity", f"{sim:.2%}", delta="High" if sim > 0.7 else "Medium" if sim > 0.4 else "Low")

        st.subheader("Find Similar Skills")
        target_skill = st.selectbox("Target skill", skills, key="target_skill")
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)
        if st.button("Find Similar Skills to Target"):
            similar = modules["embedder"].find_similar_skills(target_skill, [s for s in skills if s != target_skill], threshold=threshold, top_k=10)
            if similar:
                st.success(f"Found {len(similar)} similar skills.")
                df = pd.DataFrame(similar, columns=["Skill", "Similarity"])
                df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df)


def ner_training_step():
    st.header("Step 4: Train Custom NER Model")
    source = st.radio("Load training data from:", ["Annotations", "Upload JSON"], horizontal=True)
    training_data = None
    if source == "Annotations":
        training_data = st.session_state.get("training_annotations", [])
        if training_data:
            st.success(f"Loaded {len(training_data)} annotations from Annotation tab")
        else:
            st.warning("No annotations found. Use Annotation tab to create.")
    else:
        uploaded = st.file_uploader("Upload JSON training data", type=['json'])
        if uploaded:
            try:
                training_data = json.load(uploaded)
                st.success(f"Loaded {len(training_data)} training examples.")
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")

    if training_data:
        trainer = modules["ner_trainer"]
        spacy_data = trainer.prepare_training_data(training_data)
        n_iter = st.slider("Training iterations", 10, 100, 30, 10)
        if st.button("Start Training"):
            with st.spinner("Training custom NER..."):
                trainer.create_blank_model()
                stats = trainer.train(spacy_data, n_iter)
                st.session_state.trained_ner = trainer
                st.session_state.training_stats = stats
                st.success("Training completed!")

        if st.session_state.get("trained_ner"):
            test_text = st.text_area("Test trained NER model", height=100)
            if st.button("Predict Entities"):
                preds = st.session_state.trained_ner.predict(test_text)
                if preds:
                    st.success(f"Detected {len(preds)} skill entities:")
                    for s, start, end in preds:
                        st.write(f"- {s} (pos {start}-{end})")
                else:
                    st.info("No skill entities detected.")


def annotation_step():
    st.header("Step 5: Annotate Skills for Training")
    modules["annotator"].create_annotation_ui()


def visualization_step():
    st.header("Step 6: Visualize Extraction Results")
    if 'extraction_results' not in st.session_state:
        st.info("Extract skills first to visualize results.")
        return
    result = st.session_state.extraction_results
    vis = modules["visualizer"]

    st.plotly_chart(vis.create_category_distribution_chart(result.get('categorized_skills', {})), use_container_width=True)
    n_top = st.slider("Number of top skills to display", 5, 30, 15)
    st.plotly_chart(vis.create_top_skills_chart(result.get('all_skills', []), result.get('skill_confidence', {}), n_top), use_container_width=True)
    st.plotly_chart(vis.create_extraction_methods_chart(result.get('extraction_methods', {})), use_container_width=True)

    st.subheader("Skill Details")
    df_data = []
    skill_db = modules["extractor"].skill_db
    for skill in result.get('all_skills', []):
        category = skill_db.get_category_for_skill(skill)
        conf = result.get('skill_confidence', {}).get(skill, 0)
        df_data.append({
            "Skill": skill,
            "Category": category.replace("_", " ").title(),
            "Confidence": f"{conf:.0%}",
            "Confidence Score": conf
        })
    df = pd.DataFrame(df_data)
    categories = ['All'] + sorted(df['Category'].unique())
    selected_cat = st.selectbox("Filter by Category", categories)
    min_conf = st.slider("Minimum Confidence Score", 0.0, 1.0, 0.0, 0.1)
    filtered = df if selected_cat == 'All' else df[df['Category'] == selected_cat]
    filtered = filtered[filtered['Confidence Score'] >= min_conf]
    filtered_sorted = filtered.sort_values('Confidence Score', ascending=False)
    st.dataframe(filtered_sorted[['Skill', 'Category', 'Confidence']], use_container_width=True)
    st.caption(f"Showing {len(filtered_sorted)} of {len(df)} skills.")


def export_step():
    st.header("Step 7: Export Data")
    if 'extraction_results' not in st.session_state:
        st.info("Extract skills first to export data.")
        return
    result = st.session_state.extraction_results

    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = Exporter.to_csv(result)
        st.download_button("Download CSV", csv_data,
                           file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with col2:
        json_data = Exporter.to_json(result)
        st.download_button("Download JSON", json_data,
                           file_name=f"skills_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                           mime="application/json")
    with col3:
        report = Exporter.to_text_report(result)
        st.download_button("Download Text Report", report,
                           file_name=f"skill_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain")


def main():
    st.set_page_config(page_title="AI Skill Gap Analyzer - Milestone 2", page_icon="🎯", layout="wide")

    st.markdown("""
    <style>
    .stApp {
    background: linear-gradient(135deg, #e3f2fd 0%, #fce4ec 100%);
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
    }

    /* Sidebar with brighter blue gradient and improved text contrast */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #64b5f6 0%, #2196f3 50%, #42a5f5 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Navigation and Action Buttons - lighter gradient matching sidebar */
    .stButton > button {
        background: linear-gradient(90deg, #42a5f5, #64b5f6);
        color: #ffffff;
        border-radius: 10px;
        border: none;
        padding: 10px 18px;
        font-weight: 500;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }

    /* Hover effect for all buttons */
        .stButton > button:hover {
        background: linear-gradient(90deg, #64b5f6, #90caf9);
        color: #0d47a1;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    }

    /* Skill tag aesthetics */
    .skill-tag {
        display: inline-block;
        padding: 6px 10px;
        margin: 3px;
        border-radius: 14px;
        font-size: 14px;
        transition: 0.3s ease;
    }
    .skill-tag:hover {
        transform: scale(1.05);
        cursor: pointer;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.15);
    }
    .tech-skill {
        background-color: #bbdefb;
        color: #0d47a1;
    }
    .soft-skill {
        background-color: #f8bbd0;
        color: #880e4f;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 AI Skill Gap Analyzer")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3953/3953226.png", width=120)
        st.header("📋 Navigation")
        selected_step = st.radio("Choose Step:", steps, index=st.session_state.step_index)
        st.session_state.step_index = steps.index(selected_step)

    st.markdown(f"### Step {st.session_state.step_index + 1} of {len(steps)}: {steps[st.session_state.step_index]}")

    current_step = st.session_state.step_index
    if current_step == 0:
        text_input, doc_type = input_step()
        st.session_state['input_text'] = text_input
        st.session_state['doc_type'] = doc_type
    elif current_step == 1:
        text_input = st.session_state.get("input_text", "")
        doc_type = st.session_state.get("doc_type", "resume")
        if not text_input.strip():
            st.warning("Please complete Step 1 (Input Data) first.")
        else:
            extraction_result = extraction_step(text_input, doc_type)
            if extraction_result:
                show_summary(extraction_result)
                show_categorized_skills(extraction_result)
    elif current_step == 2:
        embedding_step()
    elif current_step == 3:
        ner_training_step()
    elif current_step == 4:
        annotation_step()
    elif current_step == 5:
        visualization_step()
    elif current_step == 6:
        export_step()

    cols = st.columns([1, 1, 10])
    with cols[0]:
        st.button("⬅️ Prev", on_click=go_prev, disabled=(current_step == 0))
    with cols[1]:
        st.button("Next ➡️", on_click=go_next, disabled=(current_step == len(steps) - 1))


if __name__ == "__main__":
    main()
