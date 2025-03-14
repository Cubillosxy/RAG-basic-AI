# dense_query.py
import click
from document_loader import load_chunks
from dense_retriever import DenseRetriever

@click.command()
@click.option('--query', prompt='Enter your query', help='Query to retrieve relevant chunks.')
@click.option('--data_dir', default='data/processed_chunks/anexo', help='Directory containing chunked .txt files.')
@click.option('--top_k', default=5, help='Number of top documents to retrieve.')
@click.option('--model_name', default='sentence-transformers/all-MiniLM-L6-v2', 
              help='Sentence Transformers model to use for embeddings.')
def main(query: str, data_dir: str, top_k: int, model_name: str):
    """
    Command-line script to load documents from a directory,
    build a dense retrieval index with SentenceTransformers + FAISS,
    and retrieve the most relevant chunks.
    """
    # 1) Load chunked documents
    documents = load_chunks(data_dir)
    if not documents:
        print("No documents found. Please check your data directory or file paths.")
        return

    # 2) Initialize the dense retriever
    retriever = DenseRetriever(documents, model_name=model_name)

    # 3) Retrieve documents matching the query
    results = retriever.retrieve(query, top_k=top_k)

    # 4) Print results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, (doc_text, score) in enumerate(results, start=1):
        # Score is dot-product similarity (0 to 1 if normalized).
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        # Show only the first 300 chars for brevity
        snippet = (doc_text[:300] + "...") if len(doc_text) > 300 else doc_text
        print(f"Snippet: {snippet}")
        print("-" * 60)

if __name__ == '__main__':
    main()
