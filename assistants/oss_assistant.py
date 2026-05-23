"""
DualAssist — Open Source Assistant (Qwen 2.5-0.5B-Instruct)

Uses Hugging Face Transformers pipeline for local inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from assistants.base import BaseAssistant
from assistants.memory import ConversationMemory
from assistants.guardrails import Guardrails
from assistants.tools import get_tool_descriptions
from observability.tracer import Tracer

logger = logging.getLogger(__name__)


class OSSAssistant(BaseAssistant):
    """
    Open-source assistant powered by Qwen 2.5-0.5B-Instruct.
    
    Loads the model via transformers pipeline and handles chat
    template formatting, generation, and context management.
    """

    def __init__(
        self,
        model_id: str = config.OSS_MODEL_ID,
        memory: Optional[ConversationMemory] = None,
        guardrails: Optional[Guardrails] = None,
        tracer: Optional[Tracer] = None,
        device_map: str = "auto",
    ):
        super().__init__(
            model_name=model_id,
            memory=memory,
            guardrails=guardrails,
            tracer=tracer,
        )
        self.model_id = model_id
        self.device_map = device_map
        self._pipeline = None

    def _load_model(self):
        """Lazy-load the model pipeline on first use."""
        if self._pipeline is not None:
            return

        logger.info(f"Loading OSS model: {self.model_id}...")
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            device_map=self.device_map,
            dtype="auto",
        )
        logger.info("OSS model loaded successfully.")

    def _generate(self, message: str, history: list[dict[str, str]]) -> str:
        """
        Generate a response using the Qwen model.
        
        Formats the conversation as chat messages and uses the
        transformers pipeline for generation.
        """
        self._load_model()

        # Build the full message list
        messages = [
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT + "\n\n" + get_tool_descriptions(),
            }
        ]

        # Add conversation history from memory
        messages.extend(history)

        # Add the current user message
        messages.append({"role": "user", "content": message})

        # Generate
        try:
            outputs = self._pipeline(
                messages,
                max_new_tokens=config.OSS_MAX_NEW_TOKENS,
                temperature=config.OSS_TEMPERATURE,
                top_p=config.OSS_TOP_P,
                do_sample=True,
            )
            # Extract the assistant's response from the generated output
            generated = outputs[0]["generated_text"]
            # The pipeline returns the full conversation; get the last assistant message
            if isinstance(generated, list):
                # Chat format: list of message dicts
                for msg in reversed(generated):
                    if msg.get("role") == "assistant":
                        return msg["content"]
                return generated[-1].get("content", str(generated[-1]))
            else:
                # Raw text format — extract after the last user message
                return str(generated)
        except Exception as e:
            logger.error(f"OSS generation error: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None
