import logging
import os
from functools import cached_property
from typing import Dict, List, Union

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("portkey", "portkey-chat-completions")
class PortkeyChatCompletion(LocalChatCompletion):
    """
    Portkey API wrapper for chat completions.

    Portkey provides a unified API gateway to multiple LLM providers.
    This implementation uses the OpenAI-compatible chat completions interface.

    Usage:
        Set the PORTKEY_API_KEY environment variable with your Portkey API key.
        Models should be specified in the format: @provider/model-name
        Examples: @openai/gpt-4.1, @anthropic/claude-3-opus-20240229

    Example:
        lm_eval --model portkey \
                --model_args model=@openai/gpt-4.1 \
                --tasks healthbench
    """

    def __init__(
        self,
        base_url="https://api.portkey.ai/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        """
        Initialize Portkey chat completion model.

        Args:
            base_url: Portkey API endpoint
            tokenizer_backend: Tokenizer to use (default: None for chat completions)
            tokenized_requests: Whether to send tokenized requests (default: False)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        eval_logger.info(f"Initialized Portkey model with model: {self.model}")
        if not self.model.startswith("@"):
            eval_logger.warning(
                f"Model '{self.model}' does not start with '@'. "
                "Portkey models should be specified as @provider/model-name (e.g., @openai/gpt-4.1)"
            )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("PORTKEY_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the PORTKEY_API_KEY environment variable."
            )
        return key

    @cached_property
    def header(self):
        """Portkey uses x-portkey-api-key header for authentication."""
        return {
            "x-portkey-api-key": f"{self.api_key}",
            "Content-Type": "application/json",
        }

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """
        Create the request payload for Portkey API.

        Portkey uses OpenAI-compatible format with some differences:
        - Uses MAX_TOKENS instead of max_tokens (handled in parent class)
        - Supports additional Portkey-specific parameters
        """
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)

        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop] if stop is not None else []

        # Build the payload
        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **gen_kwargs,
        }

        # Only add stop sequences if non-empty
        if stop:
            payload["stop"] = stop[:4]  # Limit to 4 stop sequences like OpenAI

        # Add seed if provided (not all providers support this)
        if seed is not None:
            payload["seed"] = seed

        return payload

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Parse the response from Portkey API.

        Portkey returns OpenAI-compatible responses, so we can use the same parsing.
        """
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
            res = res + tmp
        return res

    def chat_template(self, chat_template: Union[bool, str] = False):
        """Portkey doesn't require a specific chat template."""
        return ""

    def loglikelihood(self, requests, **kwargs):
        """Portkey chat completions don't support loglikelihood."""
        raise NotImplementedError(
            "Loglikelihood is not supported for Portkey chat completions."
        )
