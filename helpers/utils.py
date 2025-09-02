"""
Utils functions for the whole project
"""
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import streamlit as st
from langchain.schema import Document
from loguru import logger
from pyzotero import zotero

from clients.deep_seek import DeepSeekClient
from clients.llmclient import LLMClient
from clients.mistral import MistralClient
from helpers.constant import AVAILABLE_MODELS, AvailableModel
from helpers.rag import pdf_to_text


@st.cache_resource
def get_llm_client(model: AvailableModel) -> LLMClient:
    """
    Get the LLM client to start chatting

    :param model: Desired model
    :return:
    """

    clients = {
        "DEEPSEEK": DeepSeekClient,
        "MISTRAL": MistralClient,
    }

    args_ = {"model_name": model.name}

    return clients[model.model_family](**args_)


def get_model_type_from_input(user_input: str) -> AvailableModel:
    """
    Get LLM client from user input

    :param user_input: LLM Family
    :return: AvailableClients
    """

    try:
        return AVAILABLE_MODELS[user_input]
    except KeyError:
        raise ValueError(f"Invalid client: {user_input}")


def get_zotero_instance(library_type: str = "user"):
    """
    Init a Zotero instance
    """

    try:
        return zotero.Zotero(
            os.environ["ZOTERO_USERID"], library_type, os.environ["ZOTERO_API_KEY"]
        )
    except KeyError:
        sys.exit(
            "Zotero API key not set. Please set the environment variables ZOTERO_USERID and ZOTERO_API_KEY."
        )


def get_zotero_file(reduced: bool = False) -> Dict[str, List[str]]:
    """
    Get files on Zotero instance.
    The return dict contains the url and/or absolute paths of the files.

    :return: Dictionary of url and absolute paths
    """
    zt = get_zotero_instance()
    logger.success("Just got Zotero instance.")
    items = zt.everything(zt.items()) if not reduced else zt.items()

    files = defaultdict(list)

    for art in items:
        data = art["data"]
        link_mode = data.get("linkMode", "")

        if link_mode == "linked_file" and data.get("path"):
            files["paths"].append(data["path"])

        elif data.get("url"):
            files["url"].append(data["url"])

    return files


@st.cache_data
def zotero_files_to_text() -> List[Document]:

    files = get_zotero_file()
    texts = []
    for p in files["paths"]:
        txt_pdf = pdf_to_text(p, filetype="txt")
        if txt_pdf:
            texts.append(Document(page_content=txt_pdf))
    logger.debug(files["paths"])
    logger.success(f"Just converted {len(files['paths'])} files into text.")

    return texts
