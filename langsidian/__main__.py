import argparse
import logging
from datetime import datetime
from pathlib import Path

from langsidian import ChatBot, DocumentBase


def main(args: argparse.Namespace) -> None:
  """Run the main entry point of the script.

  Args:
      args (argparse.Namespace): Arguments from an argument parser.
  """
  bot = ChatBot(args.docs_path, args.vectorstore_path, args.document_type, args.ollama_model, args.embeddings)

  try:
    while True:
      print(f'[Assistant]: {bot.answer(input("[Prompt]: "))}')
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  logging.basicConfig(
    encoding="utf-8",
    level=logging.DEBUG,
    handlers=[logging.FileHandler(f"{datetime.now():%Y-%m-%d %H:%M:%S}.log"), logging.StreamHandler()],
    format="%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s",
  )
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
    "--docs_path",
    type=Path,
    help="Path to the directory containing the documents.",
    default=(Path.home() / "Documents" / "Obsidian").absolute(),
  )
  parser.add_argument(
    "--vectorstore_path",
    type=Path,
    help="Path to the directory where the vectorstore will be stored.",
    default=Path("docs/chroma/"),
  )
  parser.add_argument(
    "--document_type", type=DocumentBase, help="The type of documents to load.", default=DocumentBase.OBSIDIAN
  )
  parser.add_argument(
    "--ollama_model",
    type=str,
    help="Name of Ollama model to load.",
    default="mistral:7b-instruct",
    choices=["mistral:7b-instruct", "gemma:7b-instruct"],
  )

  parser.add_argument(
    "--embeddings",
    type=str,
    help="The type of embedding to load.",
    default="nomic",
    choices=["nomic", "huggingface"],
  )

  main(parser.parse_args())
