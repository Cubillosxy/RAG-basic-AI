# RAG-basic-AI
(TF-IDF/BM25) RAG Basic implementation

This repo showcases a basic Retrieval-Augmented Generation RAG system

### Getting started 

#### Installing Poetry

Poetry is a dependency management tool for Python. To install it, run the following command:

- **Linux and macOS**:

    curl -sSL https://install.python-poetry.org | python3 -

    or 

    pip3 install poetry  (or pip install poetry)

- Then run: 
  poetry install
  
- To enter the virtual environment:
  poetry shell  
  (or)  
  eval $(poetry env activate)

#### Running commands 

- poetry run pytest
- poetry run python3 some.py

#### Add packages

- poetry add package_name
- poetry add --dev package_name

(poetry docs)[https://python-poetry.org/docs/]

## Data preparation 

Put all files (pdf) in the `data/raw` folder and run:

- poetry run python3 src/data_preparation.py

Explanation: 
- This script extracts PDF text, cleans duplicated characters, and chunks the data if needed.

### Clean data: 
If you want to use OpenAI, export the API key:
- export OPENAI_API_KEY=<YOUR API KEY>

Alternatively, download Ollama to run everything locally:
- https://ollama.com/download

Then run the data prep script in local mode:

- poetry run python3 src/data_preparation.py --local

(If you have issues with the default model in `config.py`, you may switch to another model by editing `OPENAI_MODEL`.)

## Running queries

We provide two separate scripts for retrieval: `bm25_query.py` and `tfidf_query.py`. Both are located in `src/queries/`, and can be run with Poetry as follows:

Example usage for BM25:

basic: `poetry run python3 src/queries/bm25_query.py -query "Que es HistoriaCard"`

```bash
poetry run python3 src/queries/bm25_query.py \
  --query "¿Qué es HistoriaCard?" \
  --data_dir "data/processed_chunks/anexo" \
  --top_k 5
```

Example usage for TF-IDF:

basic:  `poetry run python3 src/queries/tfidf_query.py --query "Que es HistoriaCard"`
```bash
poetry run python3 src/queries/tfidf_query.py \
  --query "historial crediticio en México" \
  --data_dir "data/processed_chunks/anexo" \
  --top_k 5
```

- --query: The text you want to search for among the chunked documents.
- --data_dir: Path to the folder containing your processed .txt chunks.
- --top_k: Number of top documents to retrieve for your query.

## Data Flow

1. **PDF to Raw Text**: Place your PDFs in `data/raw` and run the data preparation script.
2. **Cleaning / Chunking**: The data preparation script can clean, split, and place .txt chunks in `data/processed_chunks`.
3. **Retrieval**: Use the unified query script `query_cli.py` to retrieve top results with either TF-IDF or BM25.
4. **(Optional) Generation**: You could feed the retrieved results into a language model to build a final RAG pipeline.

## Data folder structure

- `data/raw/` : Original PDFs
- `data/extracted/` : Text extracted directly from PDFs
- `data/processed/` : Cleaned text
- `data/processed_chunks/` : Chunked text files

## Future steps

- Compare speed or quality between TF-IDF and BM25 using an evaluation framework (e.g., custom metrics, timing analysis).
- Integrate an LLM to generate final answers from the retrieved documents.
- Extend to multi-turn conversation or advanced re-ranking.

## License

GNU

---