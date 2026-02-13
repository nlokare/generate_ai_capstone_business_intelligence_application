"""
LLM configuration and initialization module.
Handles OpenAI LLM setup and configuration.
"""

import os
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()


class LLMConfig:
    """Configuration for OpenAI LLMs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM configuration.

        Args:
            model_name: OpenAI model name (default: from env or gpt-4)
            temperature: Sampling temperature (default: from env or 0.0)
            max_tokens: Maximum tokens in response (default: from env or 2000)
            api_key: OpenAI API key (default: from env)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4")
        self.temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.0"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2000"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    def get_llm(self) -> ChatOpenAI:
        """
        Get configured ChatOpenAI instance.

        Returns:
            Configured ChatOpenAI instance
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.api_key
        )

    def get_embeddings(self) -> OpenAIEmbeddings:
        """
        Get configured OpenAI embeddings.

        Returns:
            Configured OpenAIEmbeddings instance
        """
        return OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.api_key
        )


def get_default_llm() -> ChatOpenAI:
    """
    Get default LLM instance.

    Returns:
        Default configured ChatOpenAI instance
    """
    config = LLMConfig()
    return config.get_llm()


def get_default_embeddings() -> OpenAIEmbeddings:
    """
    Get default embeddings instance.

    Returns:
        Default configured OpenAIEmbeddings instance
    """
    config = LLMConfig()
    return config.get_embeddings()
