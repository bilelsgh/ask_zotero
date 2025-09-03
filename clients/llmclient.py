import sys
from abc import ABC
from typing import Dict

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from loguru import logger
from typing_extensions import List

from helpers.constant import State

sys.path.append("..")
from helpers.rag import GenerateNode, RetrieveNode, vector_store


class LLMClient(ABC):
    def __init__(self, model_name: str, client: BaseChatModel):
        self.model_name = model_name
        self.client = client
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="input", return_messages=True
        )

        self.prompts: Dict[str, PromptTemplate] = {}
        self._init_prompts()

        # rag
        self.vs = vector_store()
        self.graph = self._build_graph()

    def _init_prompts(self):
        """
        Create prompts to use in the workflow
        """

        rag_prompt_template = """
                        Here is the history of the conversation:
                        {chat_history}

                        You may also use the following retrieved information, use it only and only if it helps you to answer to the user input, else ignore it:
                        {context}

                        User: {input}
                        Assistant:
                        """
        simple_conv_prompt_template = """
                    Here is the history of the conversation:
                    {chat_history}

                    User: {input}
                    Assistant:
                    """

        self.prompts["simple_conv"] = PromptTemplate(
            template=simple_conv_prompt_template,
            input_variables=["input", "chat_history"],
        )
        self.prompts["rag"] = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["input", "context", "chat_history"],
        )

    def _build_graph(self):
        """
        Build and compile the Retrieval-Augmented Generation (RAG) graph.

        Workflow:
        1. Retrieve: perform similarity search over the vector store
        2. Generate: pass retrieved docs into the prompt and call the LLM client
        """
        graph_builder = StateGraph(State)

        # add nodes with names
        retrieve = RetrieveNode(self.vs)
        generate = GenerateNode(
            self.client, self.memory, self.prompts["simple_conv"], self.prompts["rag"]
        )

        # Add nodes to the graph
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)

        def router(state: State):
            return "retrieve" if state.get("use_rag", False) else "generate"

        # Add edges
        graph_builder.add_conditional_edges(START, router)
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def add_docs(self, docs: List[Document]) -> None:
        self.vs.add_documents(docs)
        logger.success(f"Added {len(docs)} documents")

    def ask(self, user_input: str, use_rag: bool = False) -> str:
        """
        Ask something to the model.

        :param user_input: Prompt for the model
        :param context: Context for the prompt
        :param use_rag: If True, use the RAG pipeline instead of plain conversation
        :return: Model answer
        """

        # Run RAG graph
        logger.debug(f"Use RAG: {use_rag}")
        result = self.graph.invoke({"input": user_input, "use_rag": use_rag})

        return result["answer"]

    def reset(self):
        """
        Reset the conversation

        todo: check if we need to do something with the graph
        """

        self.memory.clear()
        logger.info("Conversation reset - memory cleared")
