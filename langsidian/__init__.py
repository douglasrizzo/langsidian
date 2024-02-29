import logging
import shutil
from enum import Enum, auto
from pathlib import Path

import re2
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import ObsidianLoader, PyPDFDirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document


class DocumentBase(Enum):
  """List of document sources."""

  OBSIDIAN = auto()
  PDF = auto()


class ChatBot:
  """A Q&A chatbot class implemented using LangChain to answer questions while performing RAG on a document base."""

  def __init__(
    self,
    docs_path: Path,
    vectorstore_db_path: Path,
    document_type: DocumentBase,
    model_type: str,
    embeddings: str = "nomic",
  ) -> None:
    """Initialize a chatbot object.

    Args:
        docs_path (Path): Path to directory containing the documents to be loaded.
        vectorstore_db_path (Path): Path where the vector store will be persisted.
        document_type (DocumentBase): Type of the document base.
        model_type (str): Type of the model.
        embeddings (str): Which embedding model to use. Defaults to "nomic".
    """
    logging.info("Loading documents")
    if document_type == DocumentBase.OBSIDIAN:
      docs = ObsidianLoader(docs_path, collect_metadata=True, encoding="UTF-8").load()
      logging.info("Loaded %d documents", len(docs))
      self.__clean_obsidian_docs(docs)
    else:
      docs = PyPDFDirectoryLoader(str(docs_path)).load()

    logging.info("Loading chunks")
    chunks = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50).split_documents(docs)
    logging.info("Loaded %d chunks", len(chunks))

    logging.info("Clearing existing vector store persist directory")
    shutil.rmtree(vectorstore_db_path, ignore_errors=True)
    logging.info("Creating vector store")
    vectordb = Chroma.from_documents(
      documents=chunks,
      embedding=HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
      if embeddings == "huggingface"
      else OllamaEmbeddings(model="nomic-embed-text"),
      persist_directory=str(vectorstore_db_path),
    )

    logging.info("Loading LLM")
    llm = Ollama(model=model_type)
    logging.info("Loading prompt template")
    with Path("templates", model_type.split(":")[0] + ".txt").open(encoding="utf-8") as f:
      template = f.read()

    qa_chain_prompt = PromptTemplate.from_template(template)
    self.qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10, "lambda_mult": 0.7}),
      return_source_documents=True,
      chain_type_kwargs={"prompt": qa_chain_prompt},
    )

  @staticmethod
  def __clean_obsidian_docs(docs: list[Document]) -> None:
    """Clean Obsidian-style hyperlinks from a list of LangChain Document objects. The process is done in-place.

    Args:
        docs (list[Document]): The list of Document objects to be cleaned.
    """
    obsidian_hyperlink_pattern = r"\[\[([^\]]*?)\|([^\[]*?)\]\]"
    for doc in docs:
      s = re2.search(obsidian_hyperlink_pattern, doc.page_content)
      if s is not None:
        new_doc = re2.sub(obsidian_hyperlink_pattern, r"\2", doc.page_content)
        doc.page_content = new_doc
      doc.page_content = doc.page_content.replace("[[", "").replace("]]", "")

  def answer(self, query: str) -> str:
    """Answer a query.

    Args:
        query (str): The query.

    Returns:
        str: The answer.
    """
    answer = self.qa_chain.invoke({"query": query})
    for idx, document in enumerate(answer["source_documents"]):
      logging.debug("Document %d: %s}", idx, document.page_content)
    return answer["result"]
