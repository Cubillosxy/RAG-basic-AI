import argparse
import os
from pathlib import Path

import pdfplumber
from openai import OpenAI
from ollama import chat, ChatResponse
from config import OPENAI_MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKED_DIR = BASE_DIR / "data" / "processed_chunks"


class PDFDataExtractor:
    """
    Reads PDFs from input_dir, extracts text, and saves each as a .txt file in output_dir (no chunking).
    """

    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initializes the PDFDataExtractor with directories.

        Args:
            input_dir (Path): Directory containing PDF files.
            output_dir (Path): Directory to save extracted text files.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extracts text content from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def save_text_file(self, text: str, output_path: Path) -> None:
        """
        Saves text content to a .txt file.

        Args:
            text (str): The text to save.
            output_path (Path): Where to save the text file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

    def process_pdfs(self) -> None:
        """
        Reads all PDF files in input_dir, extracts their text (no chunking),
        and saves the output into output_dir.
        """
        for pdf_file in self.input_dir.glob("*.pdf"):
            print(f"Processing PDF: {pdf_file.name}")
            extracted_text = self.extract_text_from_pdf(pdf_file)

            if not extracted_text.strip():
                print(f"Warning: No text extracted from {pdf_file.name}")
                continue

            text_filename = pdf_file.stem + ".txt"
            text_output_path = self.output_dir / text_filename
            self.save_text_file(extracted_text, text_output_path)
            print(f"Saved extracted text to: {text_output_path}")


class TextCleaner:
    """
    Cleans extracted text files and saves the cleaned versions in a target folder.
    """

    def __init__(self, use_local: bool = True):
        """
        Initializes the TextCleaner.
        If use_local=False, uses OpenAI (OPENAI_API_KEY required).
        """
        self.use_local = use_local
        self.openai_client = None
        self.prompt_instruction = (
            "Clean the following text by removing strange characters, "
            "duplications, and extraction errors while preserving the "
            "original structure. Return only the cleaned text.\n\n"
            "Example:\n"
            "Input:\nHHiissttoorriiaaCCaarrdd:: RReevvoolluucciioonnaannddoo\n"
            "Output:\nHistoriaCard: Revolucionando\n"
        )

        if not self.use_local:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found.")
            self.openai_client = OpenAI(api_key=self.api_key)

    def clean_text(self, text: str) -> str:
        """
        Cleans the text using either a local LLM or OpenAI.
        """
        if self.use_local:
            response: ChatResponse = chat(
                model='deepseek-r1',
                messages=[
                    {'role': 'system', 'content': self.prompt_instruction},
                    {'role': 'user', 'content': text}
                ],
                options={'temperature': 0}
            )
            content = response['message']['content']
        else:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {'role': 'system', 'content': self.prompt_instruction},
                    {'role': 'user', 'content': text}
                ],
                temperature=0
            )
            content = response.choices[0].message.content

        return content

    def chunk_text(self, text: str, max_chars: int = 200) -> list[str]:
        """
        Splits the `text` into chunks of up to `max_chars` characters each,
        ensuring words do not get cut apart.
        
        1) Split text into words.
        2) Accumulate words until adding the next word would exceed `max_chars`.
        3) Start a new chunk when we exceed `max_chars`.
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for w in words:
            # If we already have words in the chunk, add 1 extra character for the space
            length_if_added = current_length + len(w) + (1 if current_chunk else 0)
            if length_if_added <= max_chars:
                current_chunk.append(w)
                current_length = length_if_added
            else:
                # Close out the current chunk
                chunks.append(" ".join(current_chunk))
                # Start a new chunk with the current word
                current_chunk = [w]
                current_length = len(w)

        # If there's anything left over, flush it as a chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_files(self, input_dir: Path, output_dir: Path, chunk_output_dir: Path, max_chars: int = 800):
        """
        Example method:
        1) Reads all .txt files in `input_dir`.
        2) Cleans them (via `clean_text`).
        3) Saves the cleaned text to `output_dir`.
        4) Splits each cleaned file into 200-character chunks.
        5) Saves each chunk in `chunk_output_dir`.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        chunk_output_dir = Path(chunk_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_output_dir.mkdir(parents=True, exist_ok=True)

        for text_file in input_dir.glob("*.txt"):
            print(f"Cleaning text file: {text_file.name}")
            with open(text_file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # 1) Clean
            cleaned_text = self.clean_text(raw_text)

            # 2) Save cleaned
            cleaned_file_path = output_dir / text_file.name
            with open(cleaned_file_path, "w", encoding="utf-8") as cf:
                cf.write(cleaned_text)
            print(f"Saved cleaned text → {cleaned_file_path}")

            # 3) Chunk
            chunks = self.chunk_text(cleaned_text, max_chars=max_chars)

            # 4) Save each chunk
            # Create a folder for all chunks of this file (optional),
            # or save them all directly in chunk_output_dir.
            file_stem = text_file.stem
            file_chunk_dir = chunk_output_dir / file_stem
            file_chunk_dir.mkdir(parents=True, exist_ok=True)

            for i, chunk in enumerate(chunks):
                chunk_file = file_chunk_dir / f"{file_stem}_chunk_{i}.txt"
                with open(chunk_file, "w", encoding="utf-8") as cfile:
                    cfile.write(chunk)
            
            print(f"Total chunks for {text_file.name}: {len(chunks)}")
        


def main():
    parser = argparse.ArgumentParser(description="Combine PDF extraction and text cleaning.")
    parser.add_argument(
        "--local", action="store_true",
        help="Use local LLM (ollama) instead of OpenAI"
    )
    args = parser.parse_args()

    print("1) Extracting PDF text to 'data/extracted' (no chunking)...")
    extractor = PDFDataExtractor(input_dir=RAW_DIR, output_dir=EXTRACTED_DIR)
    extractor.process_pdfs()

    mode = "local" if args.local else "OpenAI"
    print(f"\n2) Cleaning extracted text using {mode} model, saving to 'data/processed'...")
    cleaner = TextCleaner(use_local=args.local)
    cleaner.process_files(EXTRACTED_DIR, PROCESSED_DIR, CHUNKED_DIR)

    print("\nAll done.")


if __name__ == "__main__":
    main()
