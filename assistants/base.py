"""
DualAssist — Base Assistant Interface

Abstract base class that defines the contract for all assistant
implementations (OSS and Frontier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from assistants.memory import ConversationMemory
from assistants.guardrails import Guardrails
from observability.tracer import Tracer


class BaseAssistant(ABC):
    """
    Abstract base class for DualAssist assistants.
    
    All assistants must implement the `_generate` method.
    The `chat` method handles the full pipeline: guardrails → generate → tools → trace.
    """

    def __init__(
        self,
        model_name: str,
        memory: Optional[ConversationMemory] = None,
        guardrails: Optional[Guardrails] = None,
        tracer: Optional[Tracer] = None,
    ):
        self.model_name = model_name
        self.memory = memory or ConversationMemory()
        self.guardrails = guardrails or Guardrails()
        self.tracer = tracer or Tracer()

    @abstractmethod
    def _generate(self, message: str, history: list[dict[str, str]]) -> str:
        """
        Generate a response given the current message and conversation history.
        
        Args:
            message: The current user message.
            history: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            The model's response text.
        """
        ...

    def chat(self, message: str) -> str:
        """
        Full chat pipeline:
        1. Input guardrail check
        2. Build context from memory
        3. Generate response
        4. Output guardrail check
        5. Tool dispatch
        6. Update memory
        7. Log trace
        
        Args:
            message: The user's message.
        
        Returns:
            The assistant's response (filtered and processed).
        """
        import uuid
        from assistants.tools import dispatch_tools

        trace_id = str(uuid.uuid4())[:8]
        self.tracer.start_trace(trace_id)

        guardrail_triggered = False
        guardrail_rule = ""
        tools_used = []

        # 1. Input guardrail
        input_check = self.guardrails.check_input(message)
        if not input_check.is_safe:
            guardrail_triggered = True
            guardrail_rule = input_check.triggered_rule
            response = input_check.filtered_content
        else:
            # 2. Build context from memory
            history = self.memory.get_context()

            # 3. Generate response
            try:
                response = self._generate(message, history)
            except Exception as e:
                response = f"I encountered an error: {str(e)}. Please try again."

            # 4. Output guardrail check
            output_check = self.guardrails.check_output(response)
            if not output_check.is_safe:
                guardrail_triggered = True
                guardrail_rule = output_check.triggered_rule
                response = output_check.filtered_content
            else:
                response = output_check.filtered_content

            # 5. Tool dispatch
            response, invocations = dispatch_tools(response)
            tools_used = [inv["tool"] for inv in invocations]

        # 6. Update memory
        self.memory.add_turn(message, response)

        # 7. Log trace
        self.tracer.end_trace(
            trace_id=trace_id,
            model=self.model_name,
            prompt=message,
            response=response,
            guardrail_triggered=guardrail_triggered,
            guardrail_rule=guardrail_rule,
            tools_used=tools_used,
        )

        return response

    def reset(self) -> None:
        """Reset conversation memory and guardrail stats."""
        self.memory.reset()
        self.guardrails.reset_stats()

    def get_status(self) -> dict:
        """Return current assistant status for the UI."""
        return {
            "model": self.model_name,
            "active_turns": self.memory.get_active_turns(),
            "total_turns": self.memory.get_turn_count(),
            "guardrail_stats": self.guardrails.get_stats(),
        }
