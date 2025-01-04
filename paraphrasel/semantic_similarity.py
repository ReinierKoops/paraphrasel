from typing import Optional
import numpy as np
from paraphrasel.util import get_language_config
from sentence_transformers import SentenceTransformer


class SemanticSimilarity:
    def __init__(self, language_code: str, override_config: Optional[dict] = None):
        config = get_language_config(language_code, override_config)
        self.model = SentenceTransformer(config["model_name"], **config["params"])

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[:1]

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def one_to_one_similarity(
        self, target_text: np.ndarray, comparison_text: np.ndarray
    ) -> float:
        target_embedding = self.get_embedding(target_text)
        comparison_embedding = self.get_embedding(comparison_text)

        return self.model.similarity(target_embedding, comparison_embedding)[0][
            0
        ].tolist()

    def one_to_many_similarity(
        self, target_text: np.ndarray, comparison_texts: np.ndarray
    ) -> list[float]:
        target_embedding = self.get_embedding(target_text)
        comparison_embeddings = self.get_embeddings(comparison_texts)

        return self.model.similarity(target_embedding, comparison_embeddings)[
            0
        ].tolist()
