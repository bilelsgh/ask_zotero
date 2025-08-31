from dataclasses import dataclass
from typing import Optional

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
        model_family="MISTRAL", name="mistral-small-latest"
    ),
}
