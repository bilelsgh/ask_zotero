from langchain_mistralai import ChatMistralAI
from loguru import logger

from clients.llmclient import LLMClient


class MistralClient(LLMClient):
    """
    Client to call Mistral API
    """

    def __init__(
        self,
        model_name: str,
    ):
        """
        Initialize the MistralClient with API key, prompt instruction, and model name.

        :param key: API key for Mistral
        :param model_name: Name of the Mistral model to use
        """
        client = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            max_retries=2,
        )
        super().__init__(model_name, client)

        logger.success("Mistral client initialized.")
