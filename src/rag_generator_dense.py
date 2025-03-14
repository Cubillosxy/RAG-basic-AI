# rag_generator_dense.py
import click
import os
import openai
import numpy as np

from ollama import chat, ChatResponse
from document_loader import load_chunks
from dense_retriever import DenseRetriever
from config import OPENAI_MODEL

def generate_answer_local(query: str, retrieved_texts: list[str]) -> str:
    """
    Uses a local model (via ollama) to generate a response based on the retrieved texts.
    """
    # Combine retrieved chunks into a single context.
    context_sections = []
    for i, text in enumerate(retrieved_texts, start=1):
        context_sections.append(f"--- Document {i} ---\n{text.strip()}\n")
    context_str = "\n".join(context_sections)

    system_prompt = (
        "You are an AI assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, respond with an appropriate disclaimer.\n"
        "If the question is in Spanish, answer in Spanish.\n"
    )
    user_prompt = (
        f"Context:\n{context_str}\n"
        f"User's question: {query}\n\n"
        "Please provide a concise, helpful answer"
    )

    # Call local LLM.
    response: ChatResponse = chat(
        model='deepseek-r1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={'temperature': 0}
    )

    content = response['message']['content'].strip()
    if "</think>" in content:
        content = content.split("</think>")[1].strip()
    return content

def generate_answer_openai(query: str, retrieved_texts: list[str]) -> str:
    """
    Uses an OpenAI model to generate a response based on the retrieved texts.
    Requires OPENAI_API_KEY environment variable.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY", None)
    if not openai.api_key:
        return "Error: OPENAI_API_KEY environment variable not set."

    # Combine retrieved chunks into a single context.
    context_sections = []
    for i, text in enumerate(retrieved_texts, start=1):
        context_sections.append(f"--- Document {i} ---\n{text.strip()}\n")
    context_str = "\n".join(context_sections)

    system_prompt = (
        "You are an AI assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, respond with an appropriate disclaimer.\n"
        "If the question is in Spanish, answer in Spanish.\n"
    )
    user_prompt = (
        f"Context:\n{context_str}\n"
        f"User's question: {query}\n\n"
        "Please provide a concise, helpful answer"
    )

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    answer = response.choices[0].message.content

    return answer

@click.command()
@click.option("--query", prompt="Enter your question", help="The user query.")
@click.option("--data_dir", default="data/processed_chunks/anexo", help="Directory with .txt chunks.")
@click.option("--top_k", default=6, help="Number of top documents to retrieve.")
@click.option("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", 
              help="Sentence Transformers model for embeddings.")
@click.option("--local", "use_local", is_flag=True, default=False, help="Use local LLM (ollama) instead of OpenAI.")
def main(query: str, data_dir: str, top_k: int, model_name: str, use_local: bool):
    """
    1) Loads chunked documents and builds a dense retriever (FAISS + Sentence Transformers).
    2) Retrieves top-k chunks for the query.
    3) Calls either a local or OpenAI-based LLM to generate the final answer.
    """
    # 1) Load chunked documents
    documents = load_chunks(data_dir)
    if not documents:
        print("No documents found. Please check your data directory or file paths.")
        return
    
    # 2) Initialize the Dense Retriever
    retriever = DenseRetriever(documents, model_name=model_name)
    
    # 3) Retrieve top-k relevant chunks
    results = retriever.retrieve(query, top_k=top_k)
    retrieved_texts = [doc_text for (doc_text, _) in results]

    # 4) Generate final answer
    if use_local:
        print("\nUsing local LLM (ollama -deepseek) for generation...\n")
        final_answer = generate_answer_local(query, retrieved_texts)
    else:
        print("\nUsing OpenAI for generation...\n")
        final_answer = generate_answer_openai(query, retrieved_texts)

    # 5) Print answer
    print("===== RAG Answer =====")
    print(final_answer)

if __name__ == "__main__":
    main()
