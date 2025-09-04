from dataclasses import dataclass
from typing import Optional

from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    input: str
    context: Optional[List[Document]]
    answer: str
    use_rag: bool


@dataclass
class AvailableModel:
    name: str
    model_family: str


AVAILABLE_MODELS = {
    # "deepseek-r1": AvailableModel(model_family="DEEPSEEK_R1", name="deepseek-chat"),
    "mistral-large": AvailableModel(
        model_family="MISTRAL", name="mistral-large-latest"
    ),
    "mistral-small": AvailableModel(
        model_family="MISTRAL", name="mistral-small-latest"
    ),
}

# Metadata for SelfQueryRetriever
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year when the article was published (e.g. 2006, 2022, 2025)",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="The author(s) who published the article",
        type="string",
    ),
    AttributeInfo(name="title", description="The title of the article", type="string"),
    AttributeInfo(
        name="abstract", description="The abstract of the article", type="string"
    ),
]
