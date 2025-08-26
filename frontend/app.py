import os
import sys

sys.path.append("../ask_zotero")
import streamlit as st
from utils import display_chat_messages

from clients.llmclient import AVAILABLE_MODELS
from config.config import set_env_vars
from helpers.rag import split_docs, web_content_loader
from helpers.utils import get_llm_client, get_model_type_from_input

set_env_vars()

# Model choice
client_choice = st.selectbox("Chose your model", list(AVAILABLE_MODELS.keys()))
model_type = get_model_type_from_input(client_choice)

# rag
docs_url = ""
l1, r1 = st.columns(2)
enable_rag = l1.toggle("Use RAG?")
if enable_rag:
    docs_url = r1.text_input("URL of documents")

# Init the client
if (
    "client" not in st.session_state
    or st.session_state.client_choice != client_choice
    or st.session_state.enable_rag != enable_rag
):
    st.session_state.enable_rag = enable_rag
    st.session_state.client = get_llm_client(
        model=model_type, key=os.environ["MISTRAL_API_KEY"], use_rag=enable_rag
    )
    if docs_url:
        splitted_docs = split_docs(docs=web_content_loader(docs_url))
        st.session_state.client.add_docs(splitted_docs)
    st.session_state.client_choice = client_choice
    st.info(f"Model `{client_choice}` selected.")

# Start the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Comment puis-je t'aider ?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.client.ask(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

display_chat_messages()
