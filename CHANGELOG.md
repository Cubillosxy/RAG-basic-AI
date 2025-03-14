# Changelog for RAG Basic AI

## [0.1.0] - 2025-03-14
### Added
- Implemented the initial Retrieval-Augmented Generation (RAG) system that uses TF-IDF and BM25 for retrieval.
- Created `bm25_query.py` and `tfidf_query.py` for querying the data using BM25 and TF-IDF methods respectively.
- Added data preparation and extraction using PDF files with `data_preparation.py`.
- Integrated text cleaning using local LLMs (Ollama) or OpenAI GPT models for cleaning and chunking data.
- Created a generation mechanism for response generation using both local and OpenAI LLMs in `rag_generator.py` and `rag_generator_dense.py`.
- Added evaluation scripts for evaluating retrieval and generation performance with test queries.
- Provided basic setup instructions and examples in `README.md`.
- First release with core features for document retrieval and generation.

### Fixed
- Fixed document chunking in `data_preparation.py` to handle large PDFs effectively.
- Addressed minor errors in chunk loading and document retrieval across different retrievers.

## [Initial] - 2025-03-11
- Initial commit with basic structure for RAG implementation, including necessary setup files, scripts, and configuration.
