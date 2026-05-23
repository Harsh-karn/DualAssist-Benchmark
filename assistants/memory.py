"""
DualAssist — Conversation Memory Manager

Implements a sliding-window buffer with optional summary compression
for maintaining short-term conversational context.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single conversation turn."""
    user: str
    assistant: str


class ConversationMemory:
    """
    Manages short-term conversational memory using a sliding window.
    
    When the buffer exceeds `window_size`, the oldest turns are dropped.
    Optionally, a summary of dropped turns is maintained to preserve
    high-level context.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.turns: list[Turn] = []
        self.summary: Optional[str] = None
        self._dropped_count: int = 0

    def add_turn(self, user_message: str, assistant_response: str) -> None:
        """Add a new conversation turn and trim if needed."""
        self.turns.append(Turn(user=user_message, assistant=assistant_response))
        
        # Trim oldest turns if we exceed the window
        while len(self.turns) > self.window_size:
            dropped = self.turns.pop(0)
            self._dropped_count += 1
            # Build a running summary of dropped turns
            self._update_summary(dropped)

    def _update_summary(self, dropped_turn: Turn) -> None:
        """Maintain a brief summary of dropped conversation context."""
        snippet = f"User asked about: {dropped_turn.user[:80]}"
        if self.summary:
            # Keep summary concise — only retain last few summaries
            lines = self.summary.split(" | ")
            lines.append(snippet)
            # Keep at most 5 summary snippets
            if len(lines) > 5:
                lines = lines[-5:]
            self.summary = " | ".join(lines)
        else:
            self.summary = snippet

    def get_context(self) -> list[dict[str, str]]:
        """
        Return the conversation context as a list of message dicts
        suitable for chat model input.
        
        If there's a summary of older turns, it's prepended as a
        system-level context note.
        """
        messages = []
        
        # Add summary of older context if available
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"[Earlier conversation summary: {self.summary}]"
            })
        
        # Add recent turns
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        
        return messages

    def get_turn_count(self) -> int:
        """Total turns including dropped ones."""
        return len(self.turns) + self._dropped_count

    def get_active_turns(self) -> int:
        """Number of turns currently in the buffer."""
        return len(self.turns)

    def reset(self) -> None:
        """Clear all memory."""
        self.turns.clear()
        self.summary = None
        self._dropped_count = 0

    def get_last_exchange(self) -> Optional[Turn]:
        """Get the most recent turn, if any."""
        return self.turns[-1] if self.turns else None
