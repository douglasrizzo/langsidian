import argparse
import logging
import shutil
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import re2
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import ObsidianLoader, PyPDFDirectoryLoader
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma

from langsidian import download_with_progress


class DocumentBase(Enum):
  """List of document sources."""

  OBSIDIAN = auto()
  PDF = auto()


class ModelType(Enum):
  """List of available models."""

  MISTRAL7B = auto()
  GEMMA7B = auto()


class ChatBot:
  """A Q&A chatbot class implemented using LangChain to answer questions while performing RAG on a document base."""

  def __init__(
    self,
    docs_path: Path,
    vectorstore_db_path: Path,
    document_type: DocumentBase,
    model_type: ModelType,
  ) -> None:
    """Initialize a chatbot object.

    Args:
        docs_path (Path): Path to directory containing the documents to be loaded.
        vectorstore_db_path (Path): Path where the vector store will be persisted.
        document_type (DocumentBase): Type of the document base.
        model_type (ModelType): Type of the model.
    """
    logging.info("Loading documents")
    if document_type == DocumentBase.OBSIDIAN:
      docs = ObsidianLoader(docs_path, collect_metadata=True, encoding="UTF-8").load()
      logging.info("Loaded %d documents", len(docs))
      obsidian_hyperlink_pattern = r"\[\[([^\]]*?)\|([^\[]*?)\]\]"
      for doc in docs:
        s = re2.search(obsidian_hyperlink_pattern, doc.page_content)
        if s is not None:
          new_doc = re2.sub(obsidian_hyperlink_pattern, r"\2", doc.page_content)
          doc.page_content = new_doc
        doc.page_content = doc.page_content.replace("[[", "").replace("]]", "")
    else:
      docs = PyPDFDirectoryLoader(str(docs_path)).load()
    logging.info("Loading chunks")
    chunks = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
    logging.info("Loaded %d chunks", len(chunks))

    logging.info("Clearing existing vector store persist directory")
    shutil.rmtree(vectorstore_db_path, ignore_errors=True)
    logging.info("Creating vector store")
    vectordb = Chroma.from_documents(
      documents=chunks, embedding=HuggingFaceEmbeddings(), persist_directory=str(vectorstore_db_path)
    )

    logging.info("Loading LLM")
    if model_type == ModelType.MISTRAL7B:
      llm_weights_path = Path("models", "mistral-7b-openorca.Q4_0.gguf")
      if not llm_weights_path.exists():
        llm_weights_path.parent.mkdir(parents=True, exist_ok=True)
        download_with_progress("https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf", llm_weights_path)
      llm = GPT4All(model=str(llm_weights_path), device="gpu")
      with Path("templates/base.txt").open(encoding="utf-8") as f:
        template = f.read()
    else:
      llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-7b-it",
        task="text-generation",
        # pipeline_kwargs={"max_new_tokens": 10},
      )
      with Path("templates/base.txt").open(encoding="utf-8") as f:
        template = f.read()

    qa_chain_prompt = PromptTemplate.from_template(template)
    self.qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=vectordb.as_retriever(search_type="mmr"),
      return_source_documents=True,
      chain_type_kwargs={"prompt": qa_chain_prompt},
    )

  def answer(self, query: str) -> str:
    """Answer a query.

    Args:
        query (str): The query.

    Returns:
        str: The answer.
    """
    return self.qa_chain.invoke({"query": query})["result"]


def main(args: argparse.Namespace) -> None:
  """Run the main entry point of the script.

  Args:
      args (argparse.Namespace): Arguments from an argument parser.
  """
  bot = ChatBot(args.docs_path, args.vectorstore_path, args.document_type, args.model_type)

  try:
    while True:
      print(bot.answer(input("[Prompt]: ")))
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
    "--model_path",
    type=Path,
    help="Path to the model file.",
    default=Path("models/my_little_llm.gguf"),
  )
  parser.add_argument(
    "--document_type", type=DocumentBase, help="The type of documents to load.", default=DocumentBase.OBSIDIAN
  )

  main(parser.parse_args())
