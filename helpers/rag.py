from typing import Dict, List

import bs4
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from helpers.constant import State

# ========== Utils ========== #


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


# ========== Graph nodes ========== #


class GenerateNode:
    def __init__(
        self,
        client: BaseChatModel,
        memory: BaseChatMemory,
        conv_prompt: PromptTemplate,
        rag_prompt: PromptTemplate,
    ):
        """
        Generate an answer to the user input based on the context (and retrieved documents).

        :return: Dictionary of the answer to the user input
        """
        self.client = client
        self.memory = memory
        self.conv_prompt = conv_prompt
        self.rag_prompt = rag_prompt

    def __call__(self, state: State) -> Dict[str, str]:
        """
        :param state: State of the conversation
        """

        logger.debug("Call generate")
        chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
        use_rag = state.get("use_rag", False)

        if use_rag:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.rag_prompt.format(
                input=state["input"], context=docs_content, chat_history=chat_history
            )
        else:
            messages = self.conv_prompt.format(
                input=state["input"], chat_history=chat_history
            )

        logger.debug(messages)
        response = self.client.invoke(messages)

        self.memory.chat_memory.add_user_message(state["input"])
        self.memory.chat_memory.add_ai_message(response.content)

        return {"answer": response.content}


class RetrieveNode:
    def __init__(self, vs: InMemoryVectorStore):
        """
        Get documents similar to the user input?

        :param vs: Vector store
        :return: Documents
        """
        self.vs = vs

    def __call__(self, state: State) -> Dict[str, List[Document]]:
        logger.debug("Call retrieve")
        docs = self.vs.similarity_search(state["input"])
        return {"context": docs}
