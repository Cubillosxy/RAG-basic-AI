import re
import click
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

from conversation_manager import ConversationHistory
from conversation_manager import generate_answer_with_history
from rag_generator import retrieve_docs

# Simple String Similarity Function
def simple_string_similarity(answer: str, expected: str) -> float:
    """
    Similarity based on overlap of words between the answer and expected answer.
    """
    ans_words = set(re.findall(r'\b\w+\b', answer.lower()))
    exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
    
    if not ans_words or not exp_words:
        return 0.0
    
    overlap = ans_words.intersection(exp_words)
    return len(overlap) / len(exp_words)

# Cosine Similarity for Relevance
def cosine_similarity_score(query: str, doc_text: str) -> float:
    """
    Computes cosine similarity between the query and document text.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query, doc_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Faithfulness check by overlap of document and answer
def faithfulness_check(answer: str, retrieved_docs: List[str]) -> float:
    """
    Checks if the answer contains information from the retrieved documents.
    Faithfulness can be defined by how much the answer overlaps with the retrieved text.
    """
    overlap_scores = []
    for doc in retrieved_docs:
        overlap_score = simple_string_similarity(answer, doc)
        overlap_scores.append(overlap_score)
    
    return np.mean(overlap_scores)  # Average overlap with all retrieved documents

# Example of evaluating a response
def evaluate_answer(query: str, answer: str, retrieved_docs: List[str], expected_answer: str) -> dict:
    """
    Evaluates the answer based on faithfulness, relevance, and correctness.
    """
    # Faithfulness
    faithfulness = faithfulness_check(answer, retrieved_docs)
    
    # Relevance (using cosine similarity between query and documents)
    relevance_scores = [cosine_similarity_score(query, doc) for doc in retrieved_docs]
    relevance = np.mean(relevance_scores)
    
    # Correctness (using string similarity to expected answer)
    correctness = simple_string_similarity(answer, expected_answer)
    
    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "correctness": correctness
    }



test_queries = [
    {
        "query": "¿Qué es HistoriaCard?",
        "expected_answer": "HistoriaCard es una fintech que ofrece soluciones para mejorar el historial crediticio y promover la educación financiera."
    },
    {
        "query": "¿Qué productos ofrece HistoriaCard?",
        "expected_answer": "HistoriaCard ofrece una tarjeta de crédito y una tarjeta de débito, y planea lanzar préstamos personales."
    },
    {
        "query": "¿Cómo puedo mejorar mi puntaje de crédito usando HistoriaCard?",
        "expected_answer": "Puedes mejorar tu puntaje de crédito utilizando la HistoriaCard de manera responsable, pagando las facturas a tiempo y usando los recursos educativos de la aplicación."
    },
    {
        "query": "¿Puedo solicitar un préstamo a través de HistoriaCard?",
        "expected_answer": "HistoriaCard planea lanzar préstamos personales en el futuro."
    },
    {
        "query": "¿HistoriaCard cobra una tarifa de mantenimiento por su tarjeta de débito?",
        "expected_answer": "No, HistoriaCard no cobra tarifas mensuales ni de mantenimiento de cuenta por su tarjeta de débito."
    }
]

"""
@click.command()
@click.option('--query', prompt='Enter your query', help='Query to retrieve relevant chunks.')
@click.option('--data_dir', default='data/processed_chunks/anexo', help='Directory containing chunked .txt files.')
@click.option('--top_k', default=5, help='Number of top documents to retrieve.')
@click.option('--model_name', default='sentence-transformers/all-MiniLM-L6-v2', 
              help='Sentence Transformers model to use for embeddings.')
"""
@click.command()
@click.option("--data_dir", default="data/processed_chunks/anexo", help="Directory with .txt chunks.")
@click.option("--top_k", default=6, help="Number of top documents to retrieve.")
@click.option("--method", default="bm25", help="Retrieval method: 'bm25' or 'tfidf'.")
def run_evaluation(data_dir: str, top_k: int, method: str):
    """
    Run evaluation on the test queries using the automated metrics.
    """
    total_tests = len(test_queries)
    total_correct = 0
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_correctness = 0.0

    conversation_history = ConversationHistory()

    for test in test_queries:
        query = test["query"]
        expected_answer = test["expected_answer"]

        # Retrieve documents using the specified retrieval method (BM25/Tf-Idf)
        retrieved_docs = retrieve_docs(query, data_dir, top_k, method)
        if not retrieved_docs:
            print(f"Error retrieving documents for query: {query}")
            continue
        
        answer = generate_answer_with_history(query, conversation_history, retrieved_docs)
        
        # Evaluate the answer
        evaluation_results = evaluate_answer(query, answer, retrieved_docs, expected_answer)

        total_faithfulness += evaluation_results["faithfulness"]
        total_relevance += evaluation_results["relevance"]
        total_correctness += evaluation_results["correctness"]
        
        # Check correctness based on string similarity to expected answer
        if evaluation_results["correctness"] >= 0.30:
            total_correct += 1

    # Compute and print average metrics
    avg_faithfulness = total_faithfulness / total_tests
    avg_relevance = total_relevance / total_tests
    avg_correctness = total_correctness / total_tests
    accuracy = total_correct / total_tests

    print("\n===== Evaluation Results =====")
    print(f"Faithfulness: {avg_faithfulness:.2f}")
    print(f"Relevance: {avg_relevance:.2f}")
    print(f"Correctness: {avg_correctness:.2f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    run_evaluation()