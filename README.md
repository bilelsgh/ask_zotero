## Chat with LLM
![Python](https://img.shields.io/badge/python-3.9.21-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A basic application to start an LLM project.

### todo
-[ ] rendre le choix du modele as a param
- [ ] gérer le "rag inutile"
- [ ] rechargement inutile a chaque fois que toggle RAG -> perte historique
-[ ] ajouter le rag par upload

### Architecture
![alt text](https://github.com/bilelsgh/chat_with_llm/blob/master/config/architecture.png)

### Run
1. Set `config/conf.yaml`
```yaml
llm_api:
  key: <your-api-key>
  model_name: <model-name> # among the ones available - check clients/llmclient.py
```

2. You can use the Streamlit interface..
```bash
streamlit run frontend/app.py
```

3. ..or simply use the main file
```bash
python main.py
```
