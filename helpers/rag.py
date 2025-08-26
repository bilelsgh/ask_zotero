import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict


def vector_store() -> InMemoryVectorStore:
    """
    Init a vector store and defining its embedding model

    :return: Vector store
    """
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vs = InMemoryVectorStore(embeddings)

    return vs


def web_content_loader(url: str) -> Document:
    """
    Load and parse web content from url

    :param url: Document url
    :return:
    """

    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()


def split_docs(
    docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into chunks

    :param docs:
    :param chunk_size:
    :param chunk_overlap:
    :return:
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    return text_splitter.split_documents(docs)
