import os
import re
import openai
import click
import time
from typing import List, Dict

from bm25_retriever import BM25Retriever
from tfidf_retriever import TfidfRetriever
from dense_retriever import DenseRetriever

from ollama import chat, ChatResponse
from config import OPENAI_MODEL
from document_loader import load_chunks

##############################
# 1) TEST QUERIES & ANSWERS (in Spanish)
##############################

TEST_SET = [
    {
        "query": "¿Qué es HistoriaCard?",
        "expected_answer": (
            "HistoriaCard es una fintech mexicana que ofrece soluciones "
            "para mejorar el historial crediticio y fomentar la educación financiera."
        )
    },
    {
        "query": "¿Qué productos ofrece HistoriaCard?",
        "expected_answer": (
            "HistoriaCard ofrece una tarjeta de crédito y una tarjeta de débito, "
            "y planea lanzar préstamos personales."
        )
    },
    {
        "query": "¿La aplicación de HistoriaCard incluye recursos de educación financiera?",
        "expected_answer": (
            "Sí, HistoriaCard cuenta con una sección de educación financiera "
            "que ofrece artículos, calculadoras y cursos en línea, recompensas"
        )

    },
    {
        "query": "¿Cuál ínea de crédito inicia que ofrece la tarjeta de crédito de HistoriaCard?",
        "expected_answer": (
            "El límite de crédito inicial es de 5,000 a 50,000 MXN."
        )
    },
    {
        "query": "¿Puedo solicitar un límite de crédito mayor a 60,000 MXN con HistoriaCard?",
        "expected_answer": (
            "No, la información oficial indica un rango de 5,000 a 50,000 MXN como límite inicial, posibilidad de aumentos"
        )
    },
    {
        "query": "¿La tarjeta de débito de HistoriaCard cobra comisión por mantenimiento de cuenta?",
        "expected_answer": (
            "No, HistoriaCard no cobra comision o comisiones mensuales ni de manejo de cuenta con la tarjeta de débito."
        )

    },
    {
        "query": "¿Qué tipo de asesoría ofrece HistoriaCard para la educación financiera?",
        "expected_answer": (
            "HistoriaCard ofrece cursos en línea y asesoría financiera educación  calculadoras, historial crédito"
        )
    },
        
    {
        "query": "¿HistoriaCard ofrece opciones de compra de acciones o ETFs en su plataforma?",
        "expected_answer": (
            "No, el contexto documentos no menciona ningún servicio de inversión en acciones ni ETF en HistoriaCard"
        )
    }
]


##############################
# 2) GENERATION HELPERS 
##############################

def generate_local(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the local LLM via ollama, measuring generation time, 
    and removes chain-of-thought if present.
    """
    start_gen = time.time()
    response: ChatResponse = chat(
        model='deepseek-r1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        options={'temperature': 0, 'max_tokens': 180},
    )
    generation_time = time.time() - start_gen

    raw_content = response['message']['content']
    if '</think>' in raw_content:
        raw_content = raw_content.split('</think>', 1)[-1]
    raw_content = raw_content.replace('<think>', '').strip()
    return raw_content, generation_time

def generate_openai(system_prompt: str, user_prompt: str) -> str:
    """
    Calls OpenAI ChatCompletion, measuring generation time.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY", None)
    if not openai.api_key:
        return "Error: No OPENAI_API_KEY set.", 0.0

    start_gen = time.time()
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    generation_time = time.time() - start_gen

    answer = response.choices[0].message.content
    return answer, generation_time


##############################
# 3) EVALUATION FUNCTION
##############################

def run_evaluation(
    retriever_method: str,
    data_dir: str,
    top_k: int,
    use_local: bool
):
    """
    Runs test queries in Spanish, measuring retrieval/generation time 
    and comparing the answers to expected references.
    """
    documents = load_chunks(data_dir)
    if not documents:
        print("No documents found in data directory. Exiting.")
        return

    # Initialize the chosen retriever
    if retriever_method.lower() == "bm25":
        retriever = BM25Retriever(documents)
    elif retriever_method.lower() == "tfidf":
        retriever = TfidfRetriever(documents)
    elif retriever_method.lower() == "dense":
        retriever = DenseRetriever(documents)
    else:
        print("Unknown retriever method. Choose: bm25, tfidf, or dense.")
        return

    total_tests = len(TEST_SET)
    total_correct = 0

    total_retrieval_time = 0.0
    total_generation_time = 0.0

    for test in TEST_SET:
        query = test["query"]
        expected = test["expected_answer"]

        print("=" * 70)
        print(f"Consulta: {query}")
        print(f"Esperado: {expected}")

        # ----- RETRIEVAL STEP -----
        start_retrieval = time.time()
        results = retriever.retrieve(query, top_k=top_k)
        retrieval_time = time.time() - start_retrieval
        total_retrieval_time += retrieval_time

        # Combine documents into a Spanish prompt
        retrieved_texts = [doc_text for (doc_text, _) in results]
        context_sections = []
        for i, text in enumerate(retrieved_texts, start=1):
            context_sections.append(f"--- Documento {i} ---\n{text.strip()}\n")
        context_str = "\n".join(context_sections)

        # ----- GENERATION STEP -----
        # We'll provide instructions in Spanish to the LLM for user-facing content:
        system_prompt = (
            "Eres un asistente de IA que responde preguntas basado en el siguiente contexto. "
            "Si el contexto no contiene la respuesta, responde con una disculpa apropiada.\n"
        )
        user_prompt = (
            f"Contexto:\n{context_str}\n"
            f"Pregunta del usuario: {query}\n\n"
            "Por favor proporciona una respuesta concisa y clara en español."
        )

        if use_local:
            answer, gen_time = generate_local(system_prompt, user_prompt)
        else:
            answer, gen_time = generate_openai(system_prompt, user_prompt)
        total_generation_time += gen_time

        # Print the result
        print(f"Respuesta: {answer}")
        print(f"(Retrieval Time: {retrieval_time:.2f}s, Generation Time: {gen_time:.2f}s)")

        # ----- Simple Overlap Scoring -----
        score = simple_string_similarity(answer.lower(), expected.lower())
        is_correct = score >= 0.30
        if is_correct:
            total_correct += 1

        print(f"Overlap Score: {score:.2f} => {'ACIERTO' if is_correct else 'FALLA'}")

    # Final summary
    accuracy = total_correct / total_tests
    avg_retrieval_time = total_retrieval_time / total_tests
    avg_generation_time = total_generation_time / total_tests

    print("\n===== RESUMEN DE EVALUACIÓN =====")
    print(f"Método de recuperación: {retriever_method}")
    print(f"LLM: {'Local (ollama)' if use_local else 'OpenAI'}")
    print(f"Total de pruebas: {total_tests}")
    print(f"Aciertos (score >= 0.3): {total_correct}")
    print(f"Exactitud: {accuracy:.2%}")
    print(f"Tiempo promedio de recuperación: {avg_retrieval_time:.2f} s")
    print(f"Tiempo promedio de generación: {avg_generation_time:.2f} s")


def simple_string_similarity(answer: str, expected: str) -> float:
    """
    Similaridad simple basada en la fracción de palabras que se superponen.
    Compara solo las palabras y convierte a minúsculas para hacer la comparación insensible al caso.
    """
    # Filtrar solo palabras alfabéticas y convertir a minúsculas
    ans_words = set(re.findall(r'\b\w+\b', answer.lower()))
    exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
    
    if not ans_words or not exp_words:
        return 0.0
    
    overlap = ans_words.intersection(exp_words)
    return len(overlap) / len(exp_words)

##############################
# 4) CLI COMMAND
##############################

@click.command()
@click.option("--retriever", default="bm25", help="Choose among: bm25, tfidf, or dense.")
@click.option("--data_dir", default="data/processed_chunks/anexo", 
              help="Folder containing the .txt chunks.")
@click.option("--top_k", default=3, help="Number of documents to retrieve.")
@click.option("--local", is_flag=True, default=False, help="Use local LLM (ollama) instead of OpenAI.")
def main(retriever: str, data_dir: str, top_k: int, local: bool):
    """
    Example usage:
      poetry run python3 evaluation.py --retriever bm25 --local
      poetry run python3 evaluation.py --retriever dense
    """
    run_evaluation(
        retriever_method=retriever,
        data_dir=data_dir,
        top_k=top_k,
        use_local=local
    )

if __name__ == "__main__":
    main()
