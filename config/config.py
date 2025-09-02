import os
from pathlib import Path

import streamlit as st
import yaml
from dotenv import load_dotenv

# == Conf file
HERE = Path(__file__).parent
CONF_PATH: Path = HERE / "conf.yaml"

if CONF_PATH.is_file():
    with open(CONF_PATH, "r") as f:
        config = yaml.safe_load(f)

# == Env var
def set_env_vars():
    load_dotenv()

    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["MISTRALAI_API_KEY"] = os.getenv("MISTRAL_API_KEY")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

    os.environ["ZOTERO_API_KEY"] = os.getenv("ZOTERO_API_KEY")
    os.environ["ZOTERO_USERID"] = os.getenv("ZOTERO_USERID")

    st.session_state.enable_rag = False


if __name__ == "__main__":
    set_env_vars()
