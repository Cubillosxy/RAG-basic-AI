# dense_retriever.py
import os
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    """
    Implements a Dense Retrieval mechanism using Sentence Transformers
    and FAISS for efficient similarity search.
    """

    def __init__(self, 
                 documents: List[str], 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        :param documents: List of text documents (chunks).
        :param model_name: Name of the Sentence Transformers model to load.
        """
        self.documents = documents
        self.model_name = model_name

        print(f"Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)

        # 2) Create embeddings for each document chunk.
        print("Generating embeddings for all documents...")
        self.doc_embeddings = self.model.encode(documents, show_progress_bar=True)

        # 3) Create a FAISS index and add the embeddings.
        # We’ll use an L2 index (IndexFlatIP) if we want cosine similarity, 
        # typically we must normalize embeddings for that. 
        # Or we can store them raw and do L2 distance. 
        # For cosine similarity with FAISS, we can:
        # - L2 normalize the embeddings, then use IndexFlatIP, or
        # - or use IndexFlatL2. 
        # We'll do a simple approach: L2 normalize + IndexFlatIP. 
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.doc_embeddings = self.doc_embeddings / (norms + 1e-10)

        d = self.doc_embeddings.shape[1]  # embedding dimension
        self.index = faiss.IndexFlatIP(d)  # dot-product index
        self.index.add(self.doc_embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k most similar documents to the query based on cosine similarity.

        :param query: The user query in raw text.
        :param top_k: Number of documents to retrieve.
        :return: A list of (document_text, similarity_score).
        """
        # 1) Encode the query
        query_vec = self.model.encode([query])
        # 2) Normalize for IP-based similarity
        query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        query_vec = query_vec / (query_norm + 1e-10)

        # 3) Search FAISS
        query_vec = query_vec.astype(np.float32)
        distances, indices = self.index.search(query_vec, top_k)

        # 'distances' are dot-product similarity scores
        # 'indices' are the positions in self.documents
        results = []
        for i, idx in enumerate(indices[0]):
            doc_text = self.documents[idx]
            score = float(distances[0][i])
            results.append((doc_text, score))

        return results
