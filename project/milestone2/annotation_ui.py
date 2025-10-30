# annotation_ui.py

import streamlit as st
import json
from datetime import datetime

class AnnotationInterface:
    """Streamlit-based Annotation UI to create training data for custom NER."""

    def __init__(self):
        if 'training_annotations' not in st.session_state:
            st.session_state.training_annotations = []
        if 'current_skills' not in st.session_state:
            st.session_state.current_skills = []

    def create_annotation_ui(self):
        st.subheader("🏷️ Skill Annotation Interface")

        st.markdown("""
        **Instructions:**
        1. Enter or paste text containing skills.
        2. Add skill annotations with start/end character indices.
        3. Save annotations, export for NER training.
        """)

        input_text = st.text_area(
            "Text to annotate:",
            height=150,
            placeholder="Example: I have experience with Python and Machine Learning."
        )

        if input_text:
            st.markdown("---")
            st.text(input_text)

            with st.form("skill_annotation_form"):
                st.markdown("**Add Skill Annotation:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    skill_text = st.text_input("Skill text")
                with col2:
                    start_pos = st.number_input("Start position", min_value=0, max_value=len(input_text), value=0)
                with col3:
                    end_pos = st.number_input("End position", min_value=0, max_value=len(input_text), value=0)

                if skill_text and start_pos < end_pos:
                    extracted = input_text[start_pos:end_pos]
                    if extracted.strip() and extracted.lower() == skill_text.strip().lower():
                        st.info(f"Preview: '{extracted}' ✓")
                    else:
                        st.warning(f"Warning: The selected text '{extracted}' does not exactly match the skill text '{skill_text}'.")

                submitted = st.form_submit_button("➕ Add Skill")
                if submitted and skill_text and start_pos < end_pos:
                    if extracted.strip() and extracted.lower() == skill_text.strip().lower():
                        st.session_state.current_skills.append({
                            "text": skill_text.strip(),
                            "start": start_pos,
                            "end": end_pos,
                            "label": "SKILL"
                        })
                        st.success(f"✅ Added skill annotation: '{skill_text.strip()}'")
                    else:
                        st.error("❌ Skill text and selected substring do not match. Please correct.")

                    st.rerun()


            if st.session_state.current_skills:
                st.markdown("**Current Annotations:**")
                for idx, skill in enumerate(st.session_state.current_skills):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{idx + 1}. **{skill['text']}** (positions {skill['start']}-{skill['end']})")
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_{idx}"):
                            st.session_state.current_skills.pop(idx)
                            st.rerun()


            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Annotation"):
                    if st.session_state.current_skills:
                        entry = {
                            "text": input_text,
                            "skills": st.session_state.current_skills.copy(),
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.training_annotations.append(entry)
                        st.session_state.current_skills = []
                        st.success(f"✅ Annotation saved! Total annotations: {len(st.session_state.training_annotations)}")
                        st.rerun()

                    else:
                        st.warning("No skills to save.")

            with col2:
                if st.button("🔄 Clear Current Annotations"):
                    st.session_state.current_skills = []
                    st.rerun()


        if st.session_state.training_annotations:
            st.markdown("---")
            st.subheader(f"📚 Training Dataset ({len(st.session_state.training_annotations)} annotations)")

            for idx, annotation in enumerate(st.session_state.training_annotations):
                with st.expander(f"Annotation {idx + 1}: {len(annotation['skills'])} skills"):
                    st.text(annotation["text"])
                    st.write("Skills annotated:")
                    for s in annotation["skills"]:
                        st.write(f"- {s['text']} (start: {s['start']}, end: {s['end']})")

            col1, col2 = st.columns(2)
            with col1:
                training_json = json.dumps(st.session_state.training_annotations, indent=2)
                st.download_button(
                    "📥 Download Training Data (JSON)",
                    data=training_json,
                    file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            # Optional: export spaCy format for direct training
            with col2:
                import ner_trainer  # assuming ner_trainer.py in same working directory
                trainer = ner_trainer.CustomNERTrainer()
                spacy_format = trainer.prepare_training_data(st.session_state.training_annotations)
                spacy_json = json.dumps(spacy_format, indent=2)
                st.download_button
