# ner_trainer.py

import random
import spacy
from spacy.training import Example
from typing import List, Tuple, Dict

class CustomNERTrainer:
    """Class for training and using a custom spaCy NER model for skill extraction."""

    def __init__(self):
        self.nlp = None
        self.ner = None

    def create_blank_model(self):
        """Initialize a blank spaCy English model and add NER component."""
        self.nlp = spacy.blank("en")
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner")
        else:
            self.ner = self.nlp.get_pipe("ner")

        self.ner.add_label("SKILL")

    def prepare_training_data(self, annotations: List[Dict]) -> List[Tuple[str, Dict]]:
        """
        Convert custom annotation data to spaCy training format.

        Annotations example:
        [
            {
                'text': 'I am a Python developer.',
                'skills': [{'start': 7, 'end': 13, 'label': 'SKILL'}]
            },
            ...
        ]

        Returns a list of tuples (text, {"entities": [(start, end, label), ...]}).
        """
        training_data = []
        for ann in annotations:
            text = ann.get("text", "")
            entities = []
            for skill in ann.get("skills", []):
                start = skill.get("start")
                end = skill.get("end")
                label = skill.get("label", "SKILL")
                entities.append((start, end, label))
            training_data.append((text, {"entities": entities}))
        return training_data

    def train(self, training_data: List[Tuple[str, Dict]], n_iter: int = 30) -> Dict:
        """
        Train the custom NER model.

        Args:
            training_data: List of (text, annotations) tuples in spaCy format.
            n_iter: Number of training iterations.

        Returns:
            A dictionary with training losses per iteration.
        """
        if not self.nlp:
            self.create_blank_model()

        # Disable other pipes to only train NER
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        training_stats = {"losses": [], "iterations": n_iter}

        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for iteration in range(n_iter):
                random.shuffle(training_data)
                losses = {}

                for text, annotations in training_data:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self.nlp.update([example], sgd=optimizer, drop=0.5, losses=losses)

                training_stats["losses"].append(losses.get("ner", 0))

        return training_stats

    def predict(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Predict skill entities in text using the trained model.

        Returns:
            List of tuples (skill_text, start_char, end_char).
        """
        if not self.nlp:
            raise ValueError("Model is not trained or loaded")

        doc = self.nlp(text)
        return [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "SKILL"]
