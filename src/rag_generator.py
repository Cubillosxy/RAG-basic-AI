import click
import os
import openai

from ollama import chat, ChatResponse
from document_loader import load_chunks
from bm25_retriever import BM25Retriever
from tfidf_retriever import TfidfRetriever
from config import OPENAI_MODEL

def retrieve_docs(query: str, data_dir: str, top_k: int, method: str = "bm25") -> list[str]:
    """
    Retrieve top-k documents from the dataset using BM25 or TF-IDF.
    """
    # 1. Load documents (chunks).
    documents = load_chunks(data_dir)
    if not documents:
        print("No documents found. Please check your data directory or file paths.")
        return []
    
    if method.lower() == "bm25":
        retriever = BM25Retriever(documents)
    else:
        retriever = TfidfRetriever(documents)
    
    results = retriever.retrieve(query, top_k=top_k)
    
    # Each element in results is (doc_text, score).
    top_texts = [doc_text for (doc_text, _) in results]
    return top_texts

def generate_answer_local(query: str, retrieved_texts: list[str]) -> str:
    """
    Uses a local model (ollama) to generate a response based on the retrieved texts.
    """
    # Combine retrieved chunks into a single string as context.
    context_sections = []
    for i, text in enumerate(retrieved_texts, start=1):
        context_sections.append(f"--- Document {i} ---\n{text.strip()}\n")
    context_str = "\n".join(context_sections)

    system_prompt = (
        "You are an AI assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, respond with an appropriate disclaimer.\n"
        "If the question is in Spanish aswer in Spanish.\n"
    )
    user_prompt = (
        f"Context:\n{context_str}\n"
        f"User's question: {query}\n\n"
        "Please provide a concise, helpful answer without include thinking things just the asnwer"
    )

    # Call the local LLM (ollama).
    response: ChatResponse = chat(
        model='deepseek-r1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={'temperature': 0, 'max_tokens': 1000}
    )

    content = response['message']['content'].strip()
    if "</think>" in content:
        content = content.split("</think>")[1].strip()

    return content

def generate_answer_openai(query: str, retrieved_texts: list[str]) -> str:
    """
    Uses an OpenAI model to generate a response based on the retrieved texts.
    Requires the OPENAI_API_KEY environment variable.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY", None)
    if not openai.api_key:
        return "Error: OPENAI_API_KEY environment variable is not set."

    # Combine retrieved chunks into a single string as context.
    context_sections = []
    for i, text in enumerate(retrieved_texts, start=1):
        context_sections.append(f"--- Document {i} ---\n{text.strip()}\n")
    context_str = "\n".join(context_sections)

    # Basic instruction prompt.
    system_prompt = (
        "You are an AI assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, respond with an appropriate disclaimer.\n"
    )
    user_prompt = (
        f"Context:\n{context_str}\n"
        f"User's question: {query}\n\n"
        "Please provide a concise, helpful answer."
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
@click.option("--method", default="bm25", help="Retrieval method: 'bm25' or 'tfidf'.")
@click.option("--local", "use_local", is_flag=True, default=False, help="Use local LLM (ollama) instead of OpenAI.")
def main(query: str, data_dir: str, top_k: int, method: str, use_local: bool):
    """
    Command to run retrieval + generation:
    1) Retrieves top-k chunks using BM25 or TF-IDF.
    2) Combines chunks into a prompt.
    3) Calls either an OpenAI model or a local LLM to generate the final answer.
    """
    retrieved_texts = retrieve_docs(query, data_dir, top_k, method)
    if not retrieved_texts:
        print("No documents retrieved. Exiting.")
        return
    
    if use_local:
        print("\nUsing local LLM (ollama deepseek-r1) for generation...\n")
        final_answer = generate_answer_local(query, retrieved_texts)
    else:
        print("\nUsing OpenAI for generation...\n")
        final_answer = generate_answer_openai(query, retrieved_texts)

    print("===== RAG Answer =====")
    print(final_answer)

if __name__ == "__main__":
    main()
