## Ask Zotero
![Python](https://img.shields.io/badge/python-3.9.21-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

_**Ask Zotero**_ is an application that enables Retrieval-Augmented Generation (RAG) directly on your Zotero library.
It ensures that your questions receive answers based on your own articles, or helps you find a paper using whatever details you remember about it.

_Powered by [Streamlit](https://github.com/streamlit/streamlit), [Pyzotero](https://github.com/urschrei/pyzotero) and [Mistral AI API](https://docs.mistral.ai/api/)._

---

### Components
#### LLM
_Ask Zotero_ uses [`mistral-large-latest`](https://docs.mistral.ai/getting-started/models/models_overview/) but we plan to add more clients so you can use your favorite LLM API or a local LLM.
#### Tokenizer
``raise I'llDoItLaterError``
#### Embedding
``raise I'llDoItLaterError``

### Set environment variables
1. Set your Mistral API key in `config/.env`
```
MISTRAL_API_KEY=""
```
2. In the same file, set your Zotero API key and Zotero UserId. Please find how to get them [here](https://pyzotero.readthedocs.io/en/latest/#getting-started-short-version).
```
ZOTERO_API_KEY=""
ZOTERO_USERID=""
```

### Run
1. Run the Streamlit application
```bash
uv run streamlit run frontend/app.py
```
