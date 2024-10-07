import os
from typing import Dict, List, Optional

try:
    import notdiamond
except ImportError:
    raise ImportError("The 'notdiamond' library is required. Please install it using 'pip install notdiamond'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.llms.litellm import LiteLLM


class NotDiamondLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)
        print("config:", self.config)

        api_key = self.config.api_key or os.getenv("NOTDIAMOND_API_KEY")
        self.client = notdiamond.NotDiamond(api_key=api_key)

        if not self.config.models:
            self.config.models = ["gpt-4o-mini", "gpt-4o"]

    def _get_model_recommendation(self, messages, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            tools (list, optional): List of tools that the model can call. Defaults to None.

        Returns:
            str: The model recommended by Not Diamond's router.
        """
        result, session_id, provider = self.client.chat.completions.route(
            messages=messages,
            models=self.config.models,
        )
        return provider.model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Litellm.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        recommended_model = self._get_model_recommendation(messages, tools)
        print("recommended_model:", recommended_model)
        self.config.model = recommended_model
        print("config:", self.config)

        litellm = LiteLLM(config=self.config)
        response = litellm.generate_response(messages, response_format, tools, tool_choice)
        return response
