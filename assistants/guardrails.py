"""
DualAssist — Safety Guardrails

Implements a layered safety system:
  - Input guardrail:  keyword blocklist + jailbreak pattern detection
  - Output guardrail: toxicity keyword scan + response filtering
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    is_safe: bool
    reason: str = ""
    filtered_content: str = ""
    triggered_rule: str = ""


class Guardrails:
    """
    Layered safety guardrail system.
    
    Pre-inference:  Checks user input for blocked content and jailbreak attempts.
    Post-inference: Checks model output for harmful/toxic content.
    """

    def __init__(self):
        self.blocked_keywords = [kw.lower() for kw in config.BLOCKED_KEYWORDS]
        self.jailbreak_patterns = [jp.lower() for jp in config.JAILBREAK_PATTERNS]
        
        # Output toxicity indicators
        self.toxic_patterns = [
            r"\b(kill|murder|assassinate)\s+(him|her|them|people|someone)\b",
            r"\b(instructions?\s+to|how\s+to|steps?\s+to)\s+(make|build|create)\s+(bomb|weapon|explosive|poison)\b",
            r"\b(racial|ethnic)\s+slur",
            r"\bsupremacy\b",
        ]
        self.toxic_regexes = [re.compile(p, re.IGNORECASE) for p in self.toxic_patterns]
        
        # Stats
        self.input_blocks = 0
        self.output_blocks = 0
        self.jailbreak_attempts = 0

    def check_input(self, user_message: str) -> GuardrailResult:
        """
        Check user input BEFORE sending to the model.
        
        Returns GuardrailResult indicating whether the input is safe.
        """
        text_lower = user_message.lower().strip()
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword in text_lower:
                self.input_blocks += 1
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Input contains blocked content",
                    filtered_content=(
                        "I'm sorry, but I can't help with that request. "
                        "It appears to involve potentially harmful content. "
                        "I'm here to help with safe and constructive tasks. "
                        "Is there something else I can assist you with?"
                    ),
                    triggered_rule="blocked_keyword",
                )
        
        # Check for jailbreak patterns
        for pattern in self.jailbreak_patterns:
            if pattern in text_lower:
                self.jailbreak_attempts += 1
                self.input_blocks += 1
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Jailbreak attempt detected",
                    filtered_content=(
                        "I notice you're trying to override my safety guidelines. "
                        "I'm designed to be helpful, harmless, and honest, and I "
                        "can't bypass my safety instructions. I'm happy to help "
                        "you with any legitimate questions or tasks!"
                    ),
                    triggered_rule="jailbreak_pattern",
                )
        
        return GuardrailResult(is_safe=True)

    def check_output(self, model_response: str) -> GuardrailResult:
        """
        Check model output AFTER generation, BEFORE showing to user.
        
        Returns GuardrailResult indicating whether the output is safe.
        """
        # Check for toxic patterns in output
        for regex in self.toxic_regexes:
            if regex.search(model_response):
                self.output_blocks += 1
                return GuardrailResult(
                    is_safe=False,
                    reason="Output contains potentially harmful content",
                    filtered_content=(
                        "I apologize, but I generated a response that may contain "
                        "inappropriate content. Let me try again with a more "
                        "appropriate answer. Could you please rephrase your question?"
                    ),
                    triggered_rule="toxic_output",
                )
        
        return GuardrailResult(is_safe=True, filtered_content=model_response)

    def get_stats(self) -> dict:
        """Return guardrail statistics."""
        return {
            "input_blocks": self.input_blocks,
            "output_blocks": self.output_blocks,
            "jailbreak_attempts": self.jailbreak_attempts,
            "total_blocks": self.input_blocks + self.output_blocks,
        }

    def reset_stats(self) -> None:
        """Reset all counters."""
        self.input_blocks = 0
        self.output_blocks = 0
        self.jailbreak_attempts = 0
