"""
DualAssist — Frontier Model Assistant (Gemini 2.0 Flash)

Uses the google-genai SDK for multi-turn conversation with
automatic history management.
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


class FrontierAssistant(BaseAssistant):
    """
    Frontier assistant powered by Google Gemini 2.0 Flash.
    
    Uses the google-genai SDK with stateless generate_content calls.
    History is managed by our own ConversationMemory for consistency
    with the OSS assistant.
    """

    def __init__(
        self,
        model_id: str = config.FRONTIER_MODEL_ID,
        memory: Optional[ConversationMemory] = None,
        guardrails: Optional[Guardrails] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(
            model_name=model_id,
            memory=memory,
            guardrails=guardrails,
            tracer=tracer,
        )
        self.model_id = model_id
        self._client = None

    def _ensure_client(self):
        """Initialize the Gemini client if not already done."""
        if self._client is not None:
            return

        if not config.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not set. Please add it to your .env file "
                "or set the environment variable."
            )

        from google import genai
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        logger.info("Gemini client initialized.")

    def _generate(self, message: str, history: list[dict[str, str]]) -> str:
        """
        Generate a response using Gemini 2.0 Flash.
        
        Builds the full conversation context and sends it as a
        single generate_content call for consistency.
        """
        self._ensure_client()
        from google.genai import types

        # Build contents list for the API
        contents = []

        # Add conversation history
        for msg in history:
            role = msg["role"]
            # Gemini uses "user" and "model" roles (not "assistant")
            if role == "assistant":
                role = "model"
            elif role == "system":
                # System messages get prepended to the first user message
                # or included via system_instruction
                continue
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        # Add current user message
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
        )

        # Generate response
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        config.SYSTEM_PROMPT + "\n\n" + get_tool_descriptions()
                    ),
                    temperature=config.FRONTIER_TEMPERATURE,
                    max_output_tokens=config.FRONTIER_MAX_TOKENS,
                ),
            )
            return response.text or "I couldn't generate a response. Please try again."
        except Exception as e:
            logger.error(f"Frontier generation error: {e}")
            raise

    @property
    def is_available(self) -> bool:
        """Check if the Gemini API key is configured."""
        return bool(config.GOOGLE_API_KEY)
