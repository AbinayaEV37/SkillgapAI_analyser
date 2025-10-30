# visualizer.py

from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


class SkillVisualizer:
    """Advanced visualizations for skills and extraction analytics using Plotly."""

    CATEGORY_DISPLAY_NAMES = {
        'programming_languages': 'Programming Languages',
        'web_frameworks': 'Web Frameworks',
        'databases': 'Databases',
        'ml_ai': 'ML / AI',
        'ml_frameworks': 'ML Frameworks',
        'cloud_platforms': 'Cloud Platforms',
        'devops_tools': 'DevOps Tools',
        'version_control': 'Version Control',
        'testing': 'Testing',
        'soft_skills': 'Soft Skills',
        'other': 'Other'
    }

    COLOR_MAP = {
        'tech': '#1976d2',      # Blue for technical skills
        'soft': '#7b1fa2',      # Purple for soft skills
        'methods': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'category_palette': px.colors.qualitative.Plotly
    }

    @staticmethod
    def create_category_distribution_chart(categorized_skills: Dict[str, List[str]]) -> go.Figure:
        """Donut chart showing skill counts per category with hover details."""
        labels = []
        values = []
        for cat, skills in categorized_skills.items():
            if skills:
                labels.append(SkillVisualizer.CATEGORY_DISPLAY_NAMES.get(cat, cat.title().replace('_', ' ')))
                values.append(len(skills))

        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.45,
            hoverinfo='label+value+percent'
        ))
        fig.update_layout(
            title='Skill Distribution by Category',
            height=450,
            margin=dict(t=40, b=40),
            legend=dict(traceorder='normal')
        )
        return fig

    @staticmethod
    def create_top_skills_chart(skills: List[str], confidence_scores: Dict[str, float], top_n: int = 15) -> go.Figure:
        """Horizontal bar chart of top-N skills by confidence."""
        top_items = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        skill_names = [s for s, _ in top_items]
        confidences = [v * 100 for _, v in top_items]
        colors = ['#1976d2' if 'soft' not in s.lower() else '#7b1fa2' for s in skill_names]

        fig = go.Figure(go.Bar(
            x=confidences[::-1], y=skill_names[::-1], orientation='h',
            marker_color=colors[::-1],
            text=[f"{c:.0f}%" for c in confidences[::-1]],
            textposition='auto',
        ))
        fig.update_layout(
            title=f"Top {top_n} Skills by Confidence",
            xaxis_title='Confidence (%)',
            yaxis_title='Skill',
            height=500,
            margin=dict(l=150, t=40)
        )
        return fig

    @staticmethod
    def create_confidence_histogram(confidence_scores: Dict[str, float]) -> go.Figure:
        """Histogram showing distribution of confidence scores."""
        values = list(confidence_scores.values())
        fig = px.histogram(values, nbins=20, labels={'value': 'Confidence Score'},
                           title='Confidence Score Distribution',
                           color_discrete_sequence=['#1976d2'])
        fig.update_layout(height=350, margin=dict(t=40, b=40))
        return fig

    @staticmethod
    def create_extraction_methods_chart(extraction_methods: Dict[str, int]) -> go.Figure:
        """Bar chart comparing number of skills detected by each method."""
        method_labels = {
            'keyword_matching': 'Keyword Matching',
            'pos_patterns': 'POS Patterns',
            'context_based': 'Context-Based',
            'ner': 'Named Entity Recognition',
            'noun_chunks': 'Noun Chunks'
        }
        labels = [method_labels.get(m, m.replace('_', ' ').title()) for m in extraction_methods.keys()]
        values = list(extraction_methods.values())
        colors = SkillVisualizer.COLOR_MAP['methods']

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=values,
            textposition='auto'
        ))
        fig.update_layout(
            title='Skills Detected by Extraction Method',
            xaxis_title='Method',
            yaxis_title='Number of Skills',
            height=400,
            margin=dict(t=40, b=40)
        )
        return fig

    @staticmethod
    def create_sunburst_chart(categorized_skills: Dict[str, List[str]]) -> go.Figure:
        """Sunburst chart showing hierarchical skill categories and counts."""
        labels = []
        parents = []
        values = []

        # Root node
        labels.append("Skills")
        parents.append("")
        values.append(0)  # root parent has 0 to auto sum

        for cat, skills in categorized_skills.items():
            # Category node
            labels.append(SkillVisualizer.CATEGORY_DISPLAY_NAMES.get(cat, cat.title().replace('_', ' ')))
            parents.append("Skills")
            values.append(len(skills))

            # Individual skills as children
            for skill in skills:
                labels.append(skill)
                parents.append(labels[-2])  # category label is the parent
                values.append(1)

        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            maxdepth=3,
            insidetextorientation='radial',
            hoverinfo="label+value+percent parent"
        ))
        fig.update_layout(title='Skill Categories and Skills Breakdown', height=600)
        return fig

    @staticmethod
    def create_skill_profile_radar(categorized_skills: Dict[str, List[str]]) -> go.Figure:
        """Radar chart contrasting soft skills and technical skill counts."""
        categories = []
        soft_count = []
        tech_count = []

        for cat in sorted(categorized_skills.keys()):
            cats = cat.replace('_', ' ').title()
            categories.append(cats)
            if cat == 'soft_skills':
                soft_count.append(len(categorized_skills[cat]))
                tech_count.append(0)
            else:
                tech_count.append(len(categorized_skills[cat]))
                soft_count.append(0)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=soft_count,
            theta=categories,
            fill='toself',
            name='Soft Skills',
            marker_color=SkillVisualizer.COLOR_MAP['soft']
        ))
        fig.add_trace(go.Scatterpolar(
            r=tech_count,
            theta=categories,
            fill='toself',
            name='Technical Skills',
            marker_color=SkillVisualizer.COLOR_MAP['tech']
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title='Skill Profile Radar Chart',
            height=450,
            margin=dict(t=40, b=40)
        )
        return fig

    @staticmethod
    def create_clustered_bar_chart(categorized_skills: Dict[str, List[str]], confidence_scores: Dict[str, float], top_n=10) -> go.Figure:
        """Clustered bar chart comparing top soft and technical skills by confidence."""
        soft_skills = [(s, confidence_scores.get(s, 0)) for s in categorized_skills.get('soft_skills', [])]
        tech_skills = []
        for cat, skills in categorized_skills.items():
            if cat != 'soft_skills':
                tech_skills.extend([(s, confidence_scores.get(s, 0)) for s in skills])

        soft_skills_sorted = sorted(soft_skills, key=lambda x: x[1], reverse=True)[:top_n]
        tech_skills_sorted = sorted(tech_skills, key=lambda x: x[1], reverse=True)[:top_n]

        fig = go.Figure(data=[
            go.Bar(
                name='Soft Skills',
                x=[s for s, _ in soft_skills_sorted],
                y=[v * 100 for _, v in soft_skills_sorted],
                marker_color=SkillVisualizer.COLOR_MAP['soft']
            ),
            go.Bar(
                name='Technical Skills',
                x=[s for s, _ in tech_skills_sorted],
                y=[v * 100 for _, v in tech_skills_sorted],
                marker_color=SkillVisualizer.COLOR_MAP['tech']
            )
        ])

        fig.update_layout(
            barmode='group',
            title=f'Top {top_n} Soft vs Technical Skills by Confidence',
            yaxis_title='Confidence (%)',
            height=500,
            margin=dict(t=40, b=40)
        )
        return fig
