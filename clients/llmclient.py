import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain import hub
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import List, TypedDict

sys.path.append("..")
from helpers.rag import vector_store


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class LLMClient(ABC):
    def __init__(self, model_name: str, use_rag: bool = False):
        self.model_name = model_name
        self.client = None
        self.use_rag = use_rag
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="input", return_messages=True
        )

        # rag
        self.vs = vector_store()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.graph = self._build_graph()

    @abstractmethod
    def ask(self, prompt: str, context: str = "") -> str:
        raise NotImplementedError

    def _init_conversation_context(self, prompt_instruction: str):
        """
        Configuration of every component needed to start the conversation.

        :param prompt_instruction: Custom prompt instruction for the conversation
        """

        if not prompt_instruction:
            prompt_template = """
            Here is the history of the conversation:
            {chat_history}

            User: {input}
            Assistant:
            """
        else:
            prompt_template = prompt_instruction
        input_variables = ["input", "chat_history"]

        prompt = PromptTemplate(
            template=prompt_template, input_variables=input_variables
        )

        self.conversation_chain = ConversationChain(
            llm=self.client, memory=self.memory, prompt=prompt, verbose=True
        )

    def _retrieve(self, state: State):
        retrieved_docs = self.vs.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )

        logger.info("Calling LLM for generate step")
        response = self.client.invoke(messages)
        return {"answer": response.content}

    def _build_graph(self):
        """
        Build and compile the Retrieval-Augmented Generation (RAG) graph.

        Workflow:
        1. Retrieve: perform similarity search over the vector store
        2. Generate: pass retrieved docs into the prompt and call the LLM client
        """
        graph_builder = StateGraph(State)

        # add nodes with names
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)

        # define edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")

        return graph_builder.compile()

    def add_docs(self, docs: List[Document]) -> None:
        self.vs.add_documents(docs)
        logger.success(f"Added {len(docs)} documents")


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
