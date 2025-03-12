# RAG-basic-AI
(TF-IDF/BM25) RAG Basic implementation


This repo showcases a basic Retrieval-Augmented Generation RAG system


### Getting started 

#### Installing Poetry

Poetry is a dependency management tool for Python. To install it, run the following command:

- linux and MacOs:

    curl -sSL https://install.python-poetry.org | python3 -

    or 

    pip3 install poetry | pip install poetry


- run : poetry install
- poetry shell  | eval $(poetry env activate)


#### Running commands 

- poetry run pytest
- poetry run python3 some.py

#### Add packages

- poetry add package_name
- poetry add --dev package_name

(poetry docs)[https://python-poetry.org/docs/]



## Data preparation 



Put all files (pdf) on raw folder and run

- poetry run python3 src/data_preparation.py


explanation; 
data preparation,  caracteres duplicados , limpiar info 

clan data , if you are going to use open ai export de api key
- EXPORT OPENAI_API_KEY=<YOUR API KEY>

download ollama to run locally 
https://ollama.com/download 


run script

- poetry run python3 src/clean_data_text.py 
- to run using ollama locally:  poetry run python3 src/clean_data_text.py  --local 

- to run with openai, set de API_KEY 
- also you have to check if you have access to de desire model (default gpt-4o) otherwise use other model on config.py  

