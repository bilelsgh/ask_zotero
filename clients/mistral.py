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
        temperature: float = 0.7,
        max_retries: int = 2,
    ):
        """
        Initialize the MistralClient with API key, prompt instruction, and model name.

        :param model_name: Name of the Mistral model to use
        :param temperature: Temperature of the Mistral model
        :param max_retries: Number of retries
        """
        client = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            max_retries=max_retries,
        )
        super().__init__(model_name, client)

        logger.success("Mistral client initialized.")
