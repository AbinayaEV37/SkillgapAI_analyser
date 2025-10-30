# skill_database.py

from typing import Dict, List

class SkillDatabase:
    """Comprehensive skill database with categorization, abbreviations, and regex patterns."""

    def __init__(self):
        self.skills = self._load_skills()
        self.abbreviations = self._load_abbreviations()
        self.context_patterns = self._load_context_patterns()

    def _load_skills(self) -> Dict[str, List[str]]:
        return {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C',
                'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust', 'Scala', 'R',
                'MATLAB', 'Perl', 'Dart', 'Shell', 'Bash', 'PowerShell'
            ],
            'web_frameworks': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django',
                'Flask', 'FastAPI', 'Spring Boot', 'Spring', 'ASP.NET', '.NET Core',
                'Ruby on Rails', 'Laravel', 'Next.js', 'Nuxt.js', 'Svelte',
                'jQuery', 'Bootstrap', 'Tailwind CSS', 'Material-UI'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra', 'Oracle',
                'SQL Server', 'SQLite', 'MariaDB', 'DynamoDB', 'Elasticsearch',
                'Firebase', 'Neo4j', 'Snowflake', 'BigQuery'
            ],
            'ml_ai': [
                'Machine Learning', 'Deep Learning', 'Neural Networks',
                'Natural Language Processing', 'NLP', 'Computer Vision',
                'Reinforcement Learning', 'Transfer Learning',
                'Feature Engineering', 'MLOps', 'Generative AI',
                'Large Language Models', 'LLM', 'CNN', 'RNN', 'LSTM',
                'Transformer', 'BERT', 'GPT'
            ],
            'ml_frameworks': [
                'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'XGBoost',
                'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn',
                'Plotly', 'NLTK', 'spaCy', 'Hugging Face', 'OpenCV'
            ],
            'cloud_platforms': [
                'AWS', 'Amazon Web Services', 'Azure', 'Microsoft Azure',
                'Google Cloud Platform', 'GCP', 'Heroku', 'DigitalOcean'
            ],
            'devops_tools': [
                'Docker', 'Kubernetes', 'Jenkins', 'GitLab CI', 'GitHub Actions',
                'CircleCI', 'Ansible', 'Terraform', 'Prometheus', 'Grafana',
                'ELK Stack', 'Datadog'
            ],
            'version_control': [
                'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN'
            ],
            'testing': [
                'Jest', 'Mocha', 'Pytest', 'JUnit', 'Selenium', 'Cypress',
                'Postman', 'JMeter'
            ],
            'soft_skills': [
                'Leadership', 'Team Management', 'Communication',
                'Problem Solving', 'Critical Thinking', 'Analytical Skills',
                'Project Management', 'Collaboration', 'Teamwork',
                'Adaptability', 'Creativity', 'Time Management'
            ]
        }

    def _load_abbreviations(self) -> Dict[str, str]:
        return {
            'ML': 'Machine Learning',
            'DL': 'Deep Learning',
            'AI': 'Artificial Intelligence',
            'NLP': 'Natural Language Processing',
            'CV': 'Computer Vision',
            'NN': 'Neural Networks',
            'CNN': 'Convolutional Neural Networks',
            'RNN': 'Recurrent Neural Networks',
            'K8s': 'Kubernetes',
            'K8S': 'Kubernetes',
            'CI/CD': 'Continuous Integration/Continuous Deployment',
            'API': 'Application Programming Interface',
            'REST': 'Representational State Transfer',
            'SQL': 'Structured Query Language',
            'OOP': 'Object-Oriented Programming',
            'TDD': 'Test-Driven Development',
            'AWS': 'Amazon Web Services',
            'GCP': 'Google Cloud Platform'
        }

    def _load_context_patterns(self) -> List[str]:
        return [
            r'experience (?:in|with) ([\w\s\+\#\.\-]+)',
            r'proficient (?:in|with|at) ([\w\s\+\#\.\-]+)',
            r'expertise (?:in|with) ([\w\s\+\#\.\-]+)',
            r'knowledge of ([\w\s\+\#\.\-]+)',
            r'skilled (?:in|at|with) ([\w\s\+\#\.\-]+)',
            r'familiar with ([\w\s\+\#\.\-]+)',
            r'(\d+)\+?\s*years? of (?:experience )?(?:in|with) ([\w\s\+\#\.\-]+)'
        ]

    def get_all_skills(self) -> List[str]:
        all_skills = []
        for skills in self.skills.values():
            all_skills.extend(skills)
        return all_skills

    def get_category_for_skill(self, skill: str) -> str:
        skill_lower = skill.lower()
        for category, skill_list in self.skills.items():
            if any(s.lower() == skill_lower for s in skill_list):
                return category
        return 'other'
