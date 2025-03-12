from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    A BM25 retriever that indexes a list of text documents
    and returns the top-k most relevant documents for a given query.
    """
    def __init__(self, documents: list[str]):
        """
        :param documents: List of raw text documents (chunks).
        """
        self.documents = documents
        
        # For BM25, we usually tokenize each document.
        self.tokenized_docs = [doc.split(" ") for doc in documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Retrieves the top-k documents most similar to the given query.
        
        :param query: Query string in raw text.
        :param top_k: Number of documents to retrieve.
        :return: A list of tuples (document_text, bm25_score).
        """
        # Tokenize the query
        query_tokens = query.split(" ")
        
        # BM25 returns a list of relevance scores for each document
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], scores[idx]))
        return results
