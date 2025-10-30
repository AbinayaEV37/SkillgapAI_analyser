import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


class Exporter:
    """Handles export of skill extraction results to CSV, JSON, and text report formats."""

    @staticmethod
    def to_csv(result: Dict, skill_db: Optional[object] = None) -> str:
        """
        Export skills to CSV string with columns: Skill, Category, Confidence, Type.

        Args:
            result: Skill extraction result dictionary.
            skill_db: Optional; instance with get_category_for_skill(skill) method.

        Returns:
            CSV string of skill data.
        """
        data = []
        if skill_db is None:
            # Fallback dummy skill DB for demo; replace with real instance when available
            skill_db = Exporter._dummy_skill_db()

        for skill in result.get("all_skills", []):
            category = skill_db.get_category_for_skill(skill)
            confidence = result.get("skill_confidence", {}).get(skill, 0)
            skill_type = "Soft Skill" if category == "soft_skills" else "Technical Skill"
            data.append({
                "Skill": skill,
                "Category": category,
                "Confidence": confidence,
                "Type": skill_type
            })
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    @staticmethod
    def to_json(result: Dict) -> str:
        """
        Export complete extraction result as formatted JSON string,
        including metadata for audit and reproducibility.
        """
        export_data = {
            "extraction_timestamp": datetime.now().isoformat(),
            "statistics": result.get("statistics", {}),
            "skills": {
                "all_skills": result.get("all_skills", []),
                "categorized_skills": result.get("categorized_skills", {}),
                "skill_confidence": result.get("skill_confidence", {})
            },
            "extraction_methods": result.get("extraction_methods", {}),
            "metadata": {  # Additional metadata placeholder
                "exported_at": datetime.now().isoformat(),
                "source": "AI Skill Gap Analyzer v1.0"
            }
        }
        return json.dumps(export_data, indent=2)

    @staticmethod
    def to_text_report(result: Dict) -> str:
        """
        Generate a human-readable multiline text report for skills and statistics.
        Suitable for PDF conversion or quick email/report sharing.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SKILL EXTRACTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Skills Extracted: {result.get('statistics', {}).get('total_skills', 0)}")
        lines.append(f"Technical Skills: {result.get('statistics', {}).get('technical_skills', 0)}")
        lines.append(f"Soft Skills: {result.get('statistics', {}).get('soft_skills', 0)}")
        lines.append("-" * 80)
        lines.append("CATEGORIZED SKILLS")
        lines.append("-" * 80)

        categorized = result.get("categorized_skills", {})
        confidence = result.get("skill_confidence", {})

        for category, skills in sorted(categorized.items()):
            if skills:
                lines.append(f"\n{category.replace('_', ' ').title()} ({len(skills)}):")
                for skill in skills:
                    conf = confidence.get(skill, 0)
                    lines.append(f"  • {skill} (Confidence: {conf:.0%})")

        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def _dummy_skill_db():
        """
        Dummy SkillDatabase stub for category lookup, to be replaced with actual DB instance.
        This is mainly for development/test fallback.
        """
        class DummyDB:
            def get_category_for_skill(self, skill):
                soft_keywords = ['leadership', 'communication', 'management', 'problem solving']
                if any(keyword in skill.lower() for keyword in soft_keywords):
                    return 'soft_skills'
                return 'technical_skills'
        return DummyDB()
