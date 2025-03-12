import click
from document_loader import load_chunks
from bm25_retriever import BM25Retriever

@click.command()
@click.option('--query', prompt='Enter your query', help='Query to retrieve relevant chunks.')
@click.option('--data_dir', default='data/processed_chunks/anexo', help='Directory containing chunked .txt files.')
@click.option('--top_k', default=5, help='Number of top documents to retrieve.')
def main(query: str, data_dir: str, top_k: int):
    """
    Command-line script to load documents from a directory,
    build a BM25 index, and retrieve the most relevant chunks.
    """
    # 1. Load the chunked documents
    documents = load_chunks(data_dir)
    
    if not documents:
        print("No documents loaded. Check your data directory or file paths.")
        return
    
    # 2. Initialize the BM25 retriever
    retriever = BM25Retriever(documents)
    results = retriever.retrieve(query, top_k=top_k)
    
    # 4. Print results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, (doc_text, score) in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Snippet: {doc_text[:300]}...")
        print("-" * 60)

if __name__ == '__main__':
    main()
