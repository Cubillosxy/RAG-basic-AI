import click
from document_loader import load_chunks
from tfidf_retriever import TfidfRetriever

@click.command()
@click.option('--query', prompt='Enter your query', help='Query to retrieve relevant chunks.')
@click.option('--data_dir', default='data/processed_chunks/anexo', help='Directory containing chunked .txt files.')
@click.option('--top_k', default=5, help='Number of top documents to retrieve.')
def main(query: str, data_dir: str, top_k: int):
    """
    Command-line script to load documents from a directory,
    build a TF-IDF index, and retrieve the most relevant chunks.
    """
    # 1. Load the chunked documents
    documents = load_chunks(data_dir)

    if not documents or all(len(doc.strip()) == 0 for doc in documents):
        print("No valid text found. Please check your chunk files.")
        return
    
    # 2. Initialize the TF-IDF retriever
    retriever = TfidfRetriever(documents)
    
    # 3. Retrieve documents matching the query
    results = retriever.retrieve(query, top_k=top_k)
    
    # 4. Print results
    print(f"\nTop {top_k} results for query: '{query}'\n")
    for rank, (doc_text, score) in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        # Show only the first 300 chars for brevity
        print(f"Doc snippet: {doc_text[:300]}...\n{'-'*60}")

if __name__ == '__main__':
    main()
