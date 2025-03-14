# RAG-basic-AI
(TF-IDF/BM25) RAG Basic implementation

This repo showcases a basic Retrieval-Augmented Generation RAG system

### Getting started 

install requirements 

`pip3 install -r requirments.txt`

or you can just use poetry

#### Installing Poetry 

Poetry is a dependency management tool for Python. [To install it](https://python-poetry.org/docs/)

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




##   Data Preparation and Basic Retrieval - (Running queries)

We provide two separate scripts for retrieval: `bm25_query.py` and `tfidf_query.py`. Both are located in `src/queries/`, and can be run with Poetry as follows:

Example usage for BM25:

basic: `poetry run python3 src/bm25_query.py --query "Que es HistoriaCard"`

```bash
poetry run python3 src/bm25_query.py --query "¿Qué es HistoriaCard?" --top_k 5
```

Example usage for TF-IDF:

basic:  `poetry run python3 src/tfidf_query.py --query "Que es HistoriaCard"`
```bash
poetry run python3 src/tfidf_query.py --query "historial crediticio en México" --top_k 5
```

- --query: The text you want to search for among the chunked documents.
- --data_dir: Path to the folder containing your processed .txt chunks.
- --top_k: Number of top documents to retrieve for your query.

## Generation Mechanism 

Example using local LLM (ollama + deepseek-r1)
- basic: poetry run python3 src/rag_generator.py --query "What is HistoriaCard and what products does it offer? --local"

```bash
poetry run python3 src/rag_generator.py --query "What is HistoriaCard and what products does it offer?" --top_k 3 --method bm25 --local
```
OpenAI mode (default)

- basic: poetry run python3 src/rag_generator.py --query "Que ofrece history card"

```bash
export OPENAI_API_KEY="sk-..."
poetry run python3 src/rag_generator.py --query "What is HistoriaCard and what products does it offer?" --top_k 3 --method bm25
```

## Enhanced Retrieval 

The script loads your chunked .txt files, builds the embeddings and FAISS index, and prints the top 5 matches for your query

- basic: poetry run python3 src/dense_query.py --query "How does HistoriaCard help improve credit history in Mexico?"

```bash
poetry run python3 src/dense_query.py --query "How does HistoriaCard help improve credit history in Mexico?" --top_k 5 --model_name "sentence-transformers/all-MiniLM-L6-v2"
```

### Generation Mechanism  with dense data
using the FAISS index we are going to produce a human output


Default mode (openAI)
- basic: poetry run python3 src/rag_generator_dense.py --query "How does HistoriaCard help improve credit history in Mexico?"
```bash
export OPENAI_API_KEY="sk-..." 
poetry run python3 rag_generator_dense.py --query "How does HistoriaCard help improve credit history in Mexico?" --top_k 3 --model_name "sentence-transformers/all-MiniLM-L6-v2"
```bash


Local mode
- basic: poetry run python3 src/rag_generator_dense.py --query "How does HistoriaCard help improve credit history in Mexico?" --local

```bash
poetry run python3 src/rag_generator_dense.py \
  --query "How does HistoriaCard help improve credit history in Mexico?" \
  --data_dir "data/processed_chunks/anexo" \
  --top_k 3 \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --local
```

## Evalution results
you can run de evaluation using the below commands:

poetry run python3 src/evaluation.py --retriever tfidf --top_k 8
poetry run python3 src/evaluation.py --retriever dense --top_k 8
poetry run python3 src/evaluation.py --top_k 8


poetry run python3 src/evaluation.py --retriever tfidf --top_k 8  --local
poetry run python3 src/evaluation.py --retriever dense --top_k 8 --local
poetry run python3 src/evaluation.py --top_k 8 --local


# Section 2 Contextual Generation and Dialogue State Management 

## 1 Contextual Generation
we use an apprach of re-ranking docs using LLMS. 

poetry run python3 dialogue/contextual_generation.py --query "Que es HistoryCard" --method tfidf
poetry run python3 dialogue/contextual_generation.py --query  "Que es HistoryCard" --rerank_only


## Conversation History
poetry run python3 src/conversation_manager.py 


## Advance evaluation 

poetry run python3 src/automate_metrics_evaluation.py --top_k 9 --method bm25
poetry run python3 src/automate_metrics_evaluation.py --top_k 9 --method tfidf



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


## Future steps

- Compare speed or quality between TF-IDF and BM25 using an evaluation framework (e.g., custom metrics, timing analysis).
- Integrate an LLM to generate final answers from the retrieved documents.
- Extend to multi-turn conversation or advanced re-ranking.

## License

GNU

---