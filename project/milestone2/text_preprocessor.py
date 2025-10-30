# text_preprocessor.py

import spacy
import streamlit as st
from typing import Dict, Any

class TextPreprocessor:
    """Text preprocessing module using spaCy with customizations for skill extraction."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = self._load_model_with_retry(model_name)
        self._customize_stop_words()

    def _load_model_with_retry(self, model_name: str):
        """Load spaCy model, downloading if necessary, with user feedback."""
        try:
            nlp = spacy.load(model_name)
            return nlp
        except OSError:
            st.warning(f"spaCy model '{model_name}' not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            return spacy.load(model_name)

    def _customize_stop_words(self):
        """Customize stop words to retain important skill tokens like single letter programming languages."""
        programming_langs = {'c', 'r', 'go', 'd', 'f'}
        # Remove relevant tokens from spaCy stop word list
        for lang in programming_langs:
            if lang in self.nlp.Defaults.stop_words:
                self.nlp.Defaults.stop_words.remove(lang)
            # Also ensure tokenizer does not treat them as stop words
            token = self.nlp.vocab[lang]
            token.is_stop = False

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text and return linguistic features needed for skill extraction.

        Returns a dictionary with keys:
          - success: bool
          - error: error message if any
          - doc: spaCy Doc object (if success)
          - noun_chunks: list of noun chunk texts
          - entities: list of (entity text, entity label)
          - sentences: list of sentence texts
        """
        if not text or not text.strip():
            return {"success": False, "error": "Input text is empty."}

        try:
            doc = self.nlp(text)
            noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            sentences = [sent.text for sent in doc.sents]

            return {
                "success": True,
                "doc": doc,
                "noun_chunks": noun_chunks,
                "entities": entities,
                "sentences": sentences,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
