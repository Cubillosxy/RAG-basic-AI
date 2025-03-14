import pytest
from src.tfidf_retriever import TfidfRetriever

@pytest.fixture
def sample_documents():
    # Return a list of sample documents for testing
    return [
        "HistoriaCard es una fintech mexicana.",
        "HistoriaCard ofrece una tarjeta de crédito.",
        "La tarjeta de crédito HistoriaCard tiene un límite de 50,000 MXN.",
        "La aplicación HistoriaCard incluye herramientas de educación financiera."
    ]

@pytest.fixture
def tfidf_retriever(sample_documents):
    # Initialize the retriever with the sample documents
    return TfidfRetriever(sample_documents)

def test_retrieve_top_k(tfidf_retriever):
    # Test that the retrieval system works as expected
    query = "tarjeta de crédito"
    top_k = 3
    results = tfidf_retriever.retrieve(query, top_k=top_k)

    assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
    assert all(isinstance(result, tuple) and len(result) == 2 for result in results), "Each result should be a tuple of (document_text, similarity_score)"
    assert results[0][0] == "HistoriaCard ofrece una tarjeta de crédito.", "The top result should match the query"

def test_empty_query(tfidf_retriever):
    # Test retrieval with an empty query
    query = ""
    top_k = 3
    results = tfidf_retriever.retrieve(query, top_k=top_k)

    assert len(results) == 0, "Expected no results for an empty query"

def test_no_matching_results(tfidf_retriever):
    # Test retrieval with a query that has no matching results
    query = "argentina"
    top_k = 3
    results = tfidf_retriever.retrieve(query, top_k=top_k)

    assert len(results) == 0, "Expected no results for a non-matching query"

def test_single_result(tfidf_retriever):
    # Test retrieval for top-1 results
    query = "fintech mexicana"
    top_k = 1
    results = tfidf_retriever.retrieve(query, top_k=top_k)

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert "HistoriaCard es una fintech mexicana." in results[0][0], "The result should contain the matching document"
