import os
import click

from rag_generator import retrieve_docs
from config import OPENAI_MODEL
import openai

class ConversationHistory:
    def __init__(self):
        """
        Initializes an empty list to store conversation history.
        Each conversation is a tuple (user_input, model_response).
        """
        self.history = []

    def add_turn(self, user_input: str, model_response: str):
        """
        Adds a new turn to the conversation history.

        :param user_input: The user's query.
        :param model_response: The model's response.
        """
        self.history.append(("user", user_input))
        self.history.append(("system", model_response))

    def get_history(self) -> str:
        """
        Returns the entire conversation history as a string, formatted for input to the model.

        :return: The conversation history formatted as a string.
        """
        history_str = ""
        for role, text in self.history:
            history_str += f"{role}: {text}\n"
        return history_str
    


def generate_answer_with_history(query: str, conversation_history: ConversationHistory, retrieved_texts: list[str]) -> str:
    """
    Generates an answer by considering the conversation history and retrieved documents.
    """
    context = conversation_history.get_history()  # Get the full conversation context
    context += f"user: {query}\n"
    
    # Combine retrieved documents into a single string as context.
    context_sections = []
    for i, text in enumerate(retrieved_texts, start=1):
        context_sections.append(f"--- Document {i} ---\n{text.strip()}\n")
    context_str = "\n".join(context_sections)

    full_context = context + "\n" + context_str

    system_prompt = (
        "You are an AI assistant that answers questions based on the provided context. "
        "If the context does not contain the answer, respond with an appropriate disclaimer.\n"
        "If the question is in Spanish, answer in Spanish.\n"
    )
    user_prompt = f"Context:\n{full_context}\nUser's question: {query}\n\nPlease provide a concise, helpful answer."

    openai.api_key = os.environ.get("OPENAI_API_KEY", None)
    if not openai.api_key:
        return "Error: OPENAI_API_KEY environment variable is not set."
    
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )
    content = response.choices[0].message.content

    conversation_history.add_turn(query, content)  
    return content



@click.command()
@click.option("--data_dir", default="data/processed_chunks/anexo", help="Directory with .txt chunks.")
@click.option("--top_k", default=6, help="Number of top documents to retrieve.")
@click.option("--method", default="bm25", help="Retrieval method: 'bm25' or 'tfidf'.")
def main(data_dir: str, top_k: int, method: str):
    """
    A loop to handle continuous conversation, with retrieval and generation,
    appending each turn to the conversation history.
    """
    # Initialize conversation history
    conversation_history = ConversationHistory()

    print("Conversation started... (Type 'exit' to quit or Ctr + c)\n")
    
    while True:
        # Accept user input
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Exiting the conversation...")
            break
        
        # Retrieve relevant documents
        retrieved_texts = retrieve_docs(user_input, data_dir, top_k, method)
        if not retrieved_texts:
            print("No documents retrieved. Exiting.")
            break
        
        final_answer = generate_answer_with_history(user_input, conversation_history, retrieved_texts)

        # Print user input and model response
        print(f"Model: {final_answer}\n")

if __name__ == "__main__":
    main()