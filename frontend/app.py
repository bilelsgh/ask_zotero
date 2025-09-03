import os
import sys

sys.path.append("../ask_zotero")
import streamlit as st
from utils import display_chat_messages

from config.config import set_env_vars
from helpers.constant import AVAILABLE_MODELS
from helpers.rag import pdf_to_text, split_docs, web_content_loader
from helpers.utils import (
    get_llm_client,
    get_model_type_from_input,
    zotero_files_to_text,
)

set_env_vars()

st.title("Ask Zotero")

with st.sidebar:
    with st.expander("Settings"):
        # Model parameters
        client_choice = st.selectbox("Chose your model", list(AVAILABLE_MODELS.keys()))
        model_temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0, value=0.3
        )
        model_type = get_model_type_from_input(client_choice)

    # Reset
    if (
        st.button("Start a new conversation", type="primary")
        and "client" in st.session_state
    ):
        st.session_state.client.reset()
        st.session_state.messages = []

    # rag
    with st.spinner("Setting up RAG..."):
        docs = zotero_files_to_text()

# Init the client
if "client" not in st.session_state or st.session_state.client_choice != client_choice:
    st.session_state.client = get_llm_client(
        model=model_type, **{"temperature": model_temperature}
    )
    if docs:
        splitted_docs = split_docs(docs=docs)
        st.session_state.client.add_docs(splitted_docs)
    st.session_state.client_choice = client_choice

# Start the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("How can I help?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.client.ask(prompt, True)
    st.session_state.messages.append({"role": "assistant", "content": response})

display_chat_messages()
