# skill_extractor.py

import re
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Optional

from skill_database import SkillDatabase
from text_preprocessor import TextPreprocessor


class SkillExtractor:
    """Multi-method skill extraction engine with normalization, categorization, and confidence scoring."""

    def __init__(self):
        self.skill_db = SkillDatabase()
        self.preprocessor = TextPreprocessor()

    def extract_skills(self, text: str, document_type: str = "resume") -> Dict:
        """
        Extracts skills from text using multiple methods:
        keyword matching, POS patterns, context regex, NER, noun chunks.
        Returns combined normalized, categorized skills and confidence scores.
        """
        preprocess_result = self.preprocessor.preprocess_text(text)
        if not preprocess_result["success"]:
            return {"success": False, "error": preprocess_result["error"]}

        doc = preprocess_result["doc"]

        # Multi-method skill extractions
        keyword_skills = self._extract_by_keyword_matching(text)
        pos_skills = self._extract_by_pos_patterns(doc)
        context_skills = self._extract_by_context_regex(text)
        ner_skills = self._extract_by_builtin_ner(preprocess_result["entities"])
        chunk_skills = self._extract_from_noun_chunks(preprocess_result["noun_chunks"])

        # Combine and deduplicate all skills
        combined_skills = self._combine_and_deduplicate([
            keyword_skills, pos_skills, context_skills, ner_skills, chunk_skills
        ])

        # Normalize skills by abbreviation expansion and cleaning
        normalized_skills = self._normalize_skills(combined_skills)

        # Categorize skills by skill database
        categorized_skills = self._categorize_skills(normalized_skills)

        # Calculate confidence scores based on detection occurrences
        skill_confidence = self._calculate_confidence(normalized_skills, [
            keyword_skills, pos_skills, context_skills, ner_skills, chunk_skills
        ])

        stats = {
            "total_skills": len(normalized_skills),
            "technical_skills": sum(len(skills) for cat, skills in categorized_skills.items() if cat != "soft_skills"),
            "soft_skills": len(categorized_skills.get("soft_skills", []))
        }

        extraction_counts = {
            "keyword_matching": len(keyword_skills),
            "pos_patterns": len(pos_skills),
            "context_based": len(context_skills),
            "ner": len(ner_skills),
            "noun_chunks": len(chunk_skills)
        }

        return {
            "success": True,
            "all_skills": normalized_skills,
            "categorized_skills": categorized_skills,
            "skill_confidence": skill_confidence,
            "extraction_methods": extraction_counts,
            "statistics": stats
        }

    def _extract_by_keyword_matching(self, text: str) -> Set[str]:
        """Extract known skills appearing as whole words in text."""
        found = set()
        lower_text = text.lower()
        for skill in self.skill_db.get_all_skills():
            pattern = r"\b" + re.escape(skill.lower()) + r"\b"
            if re.search(pattern, lower_text):
                found.add(skill)
        return found

    def _extract_by_pos_patterns(self, doc) -> Set[str]:
        """Extract skill phrases via POS patterns like ADJ+NOUN, PROPN."""
        found = set()
        tokens = list(doc)

        # ADJ + NOUN pattern
        for i in range(len(tokens) - 1):
            if tokens[i].pos_ == "ADJ" and tokens[i + 1].pos_ in ["NOUN", "PROPN"]:
                phrase = f"{tokens[i].text} {tokens[i+1].text}"
                if self._is_valid_skill(phrase):
                    found.add(phrase)

        # Proper nouns as skills
        for token in tokens:
            if token.pos_ == "PROPN" and self._is_valid_skill(token.text):
                found.add(token.text)

        return found

    def _extract_by_context_regex(self, text: str) -> Set[str]:
        """Extract skills from context regex patterns defined in the skill DB."""
        found = set()
        for pattern in self.skill_db.context_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    skill_text = match.group(len(match.groups()))
                    skill_list = self._clean_and_split_skills(skill_text)
                    for s in skill_list:
                        if self._is_valid_skill(s):
                            found.add(s)
        return found

    def _extract_by_builtin_ner(self, entities: List[Tuple[str, str]]) -> Set[str]:
        """Extract skill candidates from spaCy builtin NER entities based on label."""
        relevant_labels = {"ORG", "PRODUCT", "GPE"}
        found = set()
        for ent_text, ent_label in entities:
            if ent_label in relevant_labels and self._is_valid_skill(ent_text):
                found.add(ent_text)
        return found

    def _extract_from_noun_chunks(self, noun_chunks: List[str]) -> Set[str]:
        """Extract skills from noun chunks filtered by validity."""
        return {chunk.strip() for chunk in noun_chunks if self._is_valid_skill(chunk)}

    def _is_valid_skill(self, text: str) -> bool:
        """Check if text is a valid skill candidate."""
        if not text or len(text.strip()) < 2:
            return False
        text_clean = text.strip()
        all_skills_lower = [s.lower() for s in self.skill_db.get_all_skills()]

        if text_clean.lower() in all_skills_lower:
            return True

        for skill in self.skill_db.get_all_skills():
            if skill.lower() in text_clean.lower() or text_clean.lower() in skill.lower():
                if abs(len(skill) - len(text_clean)) <= 3:
                    return True
        return False

    def _clean_and_split_skills(self, text: str) -> List[str]:
        """Clean and split comma/semicolon-separated skill strings into individual skills."""
        skills = re.split(r"[,;|/&]|\band\b", text)
        cleaned = []
        for skill in skills:
            s = skill.strip()
            s = re.sub(r"\b(etc|and more)\b", "", s, flags=re.IGNORECASE).strip()
            if s and len(s) > 1:
                cleaned.append(s)
        return cleaned

    def _combine_and_deduplicate(self, skill_sets: List[Set[str]]) -> List[str]:
        """Combine multiple skill sets and deduplicate preserving original casing."""
        combined = set()
        for s_set in skill_sets:
            combined.update(s_set)

        unique_skills = {}
        for skill in combined:
            key = skill.lower()
            if key not in unique_skills:
                unique_skills[key] = skill
        return sorted(unique_skills.values())

    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skills by expanding known abbreviations and removing duplicates."""
        normalized = []
        seen = set()
        for skill in skills:
            expanded = self.skill_db.abbreviations.get(skill.upper(), skill)
            if expanded.lower() not in seen:
                normalized.append(expanded)
                seen.add(expanded.lower())
        return sorted(normalized)

    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills based on skill database categories."""
        categorized = defaultdict(list)
        for skill in skills:
            category = self.skill_db.get_category_for_skill(skill)
            categorized[category].append(skill)
        # Sort each category list
        for cat in categorized:
            categorized[cat].sort()
        return dict(categorized)

    def _calculate_confidence(self, skills: List[str], method_results: List[Set[str]]) -> Dict[str, float]:
        """Calculate confidence scores as fraction of methods detecting the skill."""
        confidence_scores = {}
        for skill in skills:
            count = sum(skill in s or skill.lower() in (x.lower() for x in s) for s in method_results)
            confidence_scores[skill] = round(count / len(method_results), 2)
        return confidence_scores
