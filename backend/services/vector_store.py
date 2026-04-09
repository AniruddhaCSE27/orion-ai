import json
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from backend.core.config import config

try:
    import faiss
except ImportError:
    faiss = None


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class VectorStore:
    def __init__(self, namespace: str = "default", embedding_model: Optional[str] = None):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.namespace = namespace
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.index = None
        self.documents = []
        self.enabled = faiss is not None
        self.metadata_path = DATA_DIR / f"{self.namespace}_metadata.json"
        self.index_path = DATA_DIR / f"{self.namespace}.faiss"

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.metadata_path.exists():
            try:
                self.documents = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.documents = []

        if self.enabled and self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception:
                self.index = None

    def _persist(self):
        self.metadata_path.write_text(
            json.dumps(self.documents, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        if self.enabled and self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

    def _embed_texts(self, texts):
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        embeddings = np.array(vectors, dtype="float32")
        faiss.normalize_L2(embeddings)
        return embeddings

    def add_documents(self, documents):
        if not documents:
            return

        if not self.enabled:
            self.documents.extend(documents)
            self._persist()
            return

        texts = [doc["text"] for doc in documents]
        embeddings = self._embed_texts(texts)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.documents.extend(documents)
        self._persist()

    def count_documents(self):
        return len(self.documents)

    def similarity_search(self, query: str, top_k: int = 3):
        if not query or not self.documents:
            return []

        if not self.enabled:
            results = []
            for rank, document in enumerate(self.documents[:top_k], start=1):
                item = dict(document)
                item["score"] = 1.0 / rank
                results.append(item)
            return results

        if self.index is None:
            return []

        query_embedding = self._embed_texts([query])
        limit = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, limit)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.documents[idx])
            item["score"] = float(score)
            results.append(item)
        return results
