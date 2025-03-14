import click
from typing import List, Tuple
import openai
from config import OPENAI_MODEL
from document_loader import load_chunks



class ContextualGenerator:
    """
    Handles context-aware generation by leveraging retrieved documents.
    """

    def __init__(self, data_dir: str, method: str = 'bm25'):
        self.data_dir = data_dir
        self.documents = load_chunks(data_dir)
        if method == 'bm25':
            from bm25_retriever import BM25Retriever
            self.retriever = BM25Retriever(self.documents)
        else:
            from tfidf_retriever import TfidfRetriever
            self.retriever = TfidfRetriever(self.documents)

    def initial_retrieval(self, query: str, top_k: int = 15) -> List[str]:
        """Initial retrieval of documents."""
        results = self.retriever.retrieve(query, top_k=top_k)
        return [doc for doc, _ in results]


    def generate(self, query: str, documents: List[str], top_k:int = 3) -> str:
        """
        Performs re-ranking of the retrieved documents based on LLM-generated scores,
        and returns only the top-k re-ranked documents as context for final generation.
        """

        top_k_docs = []
        # wee to adjunt cause documents is a tuple 
        for _, (doc_text, score) in enumerate(documents, start=1):
            if top_k == 0:
                break
            top_k_docs.append(doc_text)
            top_k -= 1

        context_str = "\n\n".join(top_k_docs)
        system_prompt = (
            "You are an AI assistant specialized in providing concise and accurate answers based on given context documents."
            " If the provided context does not contain enough information to answer the question, reply appropriately."
        )
        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Provide a concise and helpful answer. if the question in Spanish, answer in Spanish."
        )

        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        answer = response.choices[0].message.content.strip()
        return answer


    def rerank_documents(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Re-rank the provided documents using LLM-generated relevance scores.
        """
        scores = []
        for doc in documents:
            prompt = (
                f"Given the following document:\n{doc}\n"
                f"Evaluate how relevant this document is for answering the query: '{query}'. "
                "Provide a numeric relevance score between 0 (irrelevant) and 1 (highly relevant). Respond only with the score."
            )

            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Evaluate document relevance numerically."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            score_str = response.choices[0].message.content.strip()
            try:
                score = float(score_str)
            except ValueError:
                score = 0.0

            scores.append(score)

        scored_documents = list(zip(documents, scores))
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents
    

def contextual_generate(query: str, data_dir: str, top_k: int = 5, method: str = 'bm25', return_reranked_docs: bool = False) -> str:
    """
    Main function for contextual generation using retrieval and re-ranking.
    
    :param query: The query string to retrieve and generate responses.
    :param data_dir: The directory containing processed documents.
    :param top_k: Number of top documents to retrieve.
    :param method: Retrieval method ('bm25' or 'tfidf').
    :param return_reranked_docs: Flag to return only the reranked documents (True) or generate a response (False).
    :return: Either a string of the final generated answer or a list of reranked documents.
    """
    gen = ContextualGenerator(data_dir=data_dir, method=method)
    initial_docs = gen.initial_retrieval(query, top_k=top_k + 5)  # Get a few extra docs for re-ranking
    reranked_docs_with_scores = gen.rerank_documents(query, initial_docs)
    
    if return_reranked_docs:
        return reranked_docs_with_scores[:top_k]

    final_answer = gen.generate(query, reranked_docs_with_scores, top_k=top_k)
    return final_answer

@click.command()
@click.option("--query", prompt="Enter your query", help="Query for contextual generation.")
@click.option("--data_dir", default="data/processed_chunks/anexo", help="Data directory containing chunks.")
@click.option("--method", default="bm25", help="Retrieval method (bm25/tfidf)")
@click.option("--top_k", default=6, help="Number of top documents to retrieve.")
@click.option("--rerank_only", is_flag=True, default=False, help="If set, only returns reranked documents without generating a response.")
def main(query, data_dir, method, top_k, rerank_only):
    response = contextual_generate(query, data_dir,top_k,method=method, return_reranked_docs=rerank_only)
    
    if rerank_only:
        print("===== Reranked Documents =====")
        for rank, (doc_text, score) in enumerate(response, start=1):
            print(f"Rank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"Doc snippet: {doc_text[:600]}...\n{'-'*60}")
    else:
        print("===== Contextual Generation Answer =====")
        print(response)

main()
