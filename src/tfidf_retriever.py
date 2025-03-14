from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class TfidfRetriever:
    """
    A simple TF-IDF retriever that indexes a list of text documents
    and returns the top-k most relevant documents for a given query.
    """
    def __init__(self, documents: list[str]):
        """
        Initializes the TF-IDF retriever and computes
        the TF-IDF vectors for all documents.
        
        :param documents: List of raw text documents.
        """
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        self.similarity_threshold = 0.1

    def retrieve(self, query: str, top_k: int = 6) -> list[tuple[str, float]]:
        """
        Retrieves the top-k documents most similar to the given query.
        
        :param query: Query string.
        :param top_k: Number of documents to retrieve.
        :return: A list of tuples (document_text, similarity_score).
        """
        # Check if the query is empty
        if not query.strip():
            return []  # Return an empty list if the query is empty
        
        # Transform query into the same TF-IDF space
        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.doc_vectors).flatten()

        # Filter results by similarity threshold
        top_indices = [i for i in range(len(similarities)) if similarities[i] >= self.similarity_threshold]

        # If no document is above the threshold, return empty
        if not top_indices:
            return []

        # Get top-k relevant results based on similarity scores
        top_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            doc_text = self.documents[idx]
            sim_score = similarities[idx]
            results.append((doc_text, sim_score))
        
        return results
