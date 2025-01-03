import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from semanticmatch.load_config import get_language_config


class SemanticSimilarity:
    def __init__(self, language_code: str):
        config = get_language_config(language_code)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], **config["tokenizer_params"]
        )
        self.model = AutoModel.from_pretrained(config["model_name"])

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return self.cosine_similarity(emb1, emb2)
