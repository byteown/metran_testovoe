import numpy as np
from sentence_transformers import SentenceTransformer

from fin_agent.rag.chunks import Chunk


class DocumentIndex:
    def __init__(self, chunks: list[Chunk], model_name: str) -> None:
        self._chunks = chunks
        self._model = SentenceTransformer(model_name)
        texts = [f"{c.breadcrumb}\n{c.text}" for c in chunks]
        self._matrix = self._model.encode(texts, normalize_embeddings=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    def search(
            self, query: str, top_k: int, min_score: float
    ) -> list[tuple[Chunk, float]]:
        vector = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._matrix @ vector
        order = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunks[i], float(scores[i])) for i in order if scores[i] >= min_score
        ]
