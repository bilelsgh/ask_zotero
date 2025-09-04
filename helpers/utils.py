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
def get_llm_client(model: AvailableModel, **args) -> LLMClient:
    """
    Get the LLM client to start chatting

    :param model: Desired model
    :return:
    """

    clients = {
        "DEEPSEEK": DeepSeekClient,
        "MISTRAL": MistralClient,
    }

    args.update({"model_name": model.name})

    return clients[model.model_family](**args)


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


@st.cache_data
def zotero_files_to_text(reduced: bool = False) -> List[Document]:
    """
    Get files on Zotero instance.
    The return dict contains the url and/or absolute paths of the files.

    :return: Dictionary of url and absolute paths
    """
    zt = get_zotero_instance()
    logger.success("Just got Zotero instance.")
    items = zt.everything(zt.items()) if not reduced else zt.items()

    docs = []

    for art in items:
        data = art["data"]
        link_mode = data.get("linkMode", "")

        # get file location
        if link_mode == "linked_file" and data.get("path"):
            path = data["path"]
            txt_pdf = pdf_to_text(path, filetype="txt")

            # metadata
            metadata = {
                "author": ",".join(
                    [
                        f"{c['firstName']} {c['lastName']}"
                        for c in data.get("creators", [])
                        if c["creatorType"] == "author"
                    ]
                ),
                "year": data.get("date", ""),
                "title": data.get("title", ""),
                "abstract": data.get("abstractNote", ""),
            }

            if txt_pdf:
                docs.append(
                    Document(
                        page_content=txt_pdf,
                        metadata=metadata,
                    )
                )

        elif data.get("url"):
            url = data["url"]
            continue

    logger.success(f"Just converted {len(docs)} files into text.")
    return docs
