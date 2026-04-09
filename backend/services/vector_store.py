import numpy as np
from openai import OpenAI
from typing import Optional

from backend.core.config import config

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    def __init__(self, embedding_model: Optional[str] = None):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.index = None
        self.documents = []
        self.enabled = faiss is not None

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
            return

        texts = [doc["text"] for doc in documents]
        embeddings = self._embed_texts(texts)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.documents.extend(documents)

    def similarity_search(self, query: str, top_k: int = 3):
        if not query or not self.documents:
            return []

        if not self.enabled:
            return self.documents[:top_k]

        if self.index is None:
            return []

        query_embedding = self._embed_texts([query])
        limit = min(top_k, len(self.documents))
        _, indices = self.index.search(query_embedding, limit)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.documents[idx])
        return results
