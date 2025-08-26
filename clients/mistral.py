from langchain_mistralai import ChatMistralAI
from loguru import logger

from clients.llmclient import LLMClient


class MistralClient(LLMClient):
    """
    Client to call Mistral API
    """

    def __init__(
        self,
        key: str,
        model_name: str,
        prompt_instruction: str = "",
        use_rag: bool = False,
    ):
        """
        Initialize the MistralClient with API key, prompt instruction, and model name.

        :param key: API key for Mistral
        :param prompt_instruction: Custom prompt instruction for the conversation
        :param model_name: Name of the Mistral model to use
        """
        super().__init__(model_name, use_rag)
        self.client = ChatMistralAI(
            model=model_name,
            temperature=0,
            max_retries=2,
        )

        self._init_conversation_context(prompt_instruction)
        logger.success("Mistral client initialized.")

    def ask(self, user_input: str, context: str = "") -> str:
        """
        Ask something to the model.

        :param user_input: Prompt for the model
        :param context: Context for the prompt
        :param use_rag: If True, use the RAG pipeline instead of plain conversation
        :return: Model answer
        """
        if self.use_rag:
            logger.info(f"About to use RAG pipeline.")

            # Run RAG graph
            result = self.graph.invoke({"question": user_input})
            answer = result["answer"]

            # update memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(answer)

            return answer

        else:
            # Normal conversation mode
            logger.info(f"About to plain conversation.")
            return self.conversation_chain.run(
                {"input": user_input, "context": context}
            )
