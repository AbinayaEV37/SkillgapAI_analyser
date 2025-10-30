# bert_embedder.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple

class SentenceBERTEmbedder:
    """Generate and manage Sentence-BERT embeddings plus similarity computations."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize Sentence-BERT model, with retry on failure."""
        try:
            self.model = SentenceTransformer(model_name)
            self._embedding_cache = {}
        except Exception as e:
            st.error(f"Error loading Sentence-BERT model: {e}")
            st.info("Try reinstalling with 'pip install sentence-transformers'.")
            raise e

    def encode_skills(self, skills: List[str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for a list of skills; caches results."""
        uncached = [s for s in skills if s not in self._embedding_cache]
        if uncached:
            new_embeddings = self.model.encode(uncached, show_progress_bar=True)
            for skill, emb in zip(uncached, new_embeddings):
                self._embedding_cache[skill] = emb
        return {skill: self._embedding_cache[skill] for skill in skills}

    def compute_similarity(self, skill1: str, skill2: str) -> float:
        """Compute cosine similarity between two skill embeddings."""
        emb1 = self._get_embedding(skill1)
        emb2 = self._get_embedding(skill2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return float(sim)

    def compute_similarity_matrix(self, skills1: List[str], skills2: List[str]) -> np.ndarray:
        """Compute cosine similarity matrix between two lists of skills."""
        emb1 = self.model.encode(skills1)
        emb2 = self.model.encode(skills2)
        sim_matrix = cosine_similarity(emb1, emb2)
        return sim_matrix

    def find_similar_skills(self, target_skill: str, skill_list: List[str], 
                            threshold: float = 0.7, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find skills in skill_list semantically similar to target_skill."""
        target_emb = self._get_embedding(target_skill)
        similarities = []
        for skill in skill_list:
            if skill.lower() != target_skill.lower():
                emb = self._get_embedding(skill)
                sim = cosine_similarity([target_emb], [emb])[0][0]
                if sim >= threshold:
                    similarities.append((skill, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _get_embedding(self, skill: str):
        """Retrieve or compute embedding for a single skill."""
        if skill not in self._embedding_cache:
            emb = self.model.encode([skill])[0]
            self._embedding_cache[skill] = emb
        return self._embedding_cache[skill]
