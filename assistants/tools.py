"""
DualAssist — Tool Definitions & Dispatcher

Provides lightweight tools the assistant can invoke:
  - calculator: safe math evaluation
  - datetime:   current date/time
  - web_search: stub search (no API cost)
"""

from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime, timezone


# ──────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────

# Safe operators for the calculator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only safe operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.func.id]
            if callable(func):
                args = [_safe_eval_node(arg) for arg in node.args]
                return func(*args)
            else:
                return func  # It's a constant like pi or e
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            val = _SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"Unsupported name: {node.id}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def tool_calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Examples:
        "2 + 3 * 4"  -> "14"
        "sqrt(144)"   -> "12.0"
        "2 ** 10"     -> "1024"
    """
    try:
        # Clean the expression
        expr = expression.strip()
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree)
        # Format nicely
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(round(result, 8))
    except Exception as e:
        return f"Calculator error: {e}"


def tool_datetime() -> str:
    """Return the current date, time, and timezone."""
    now = datetime.now()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Current local time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}\n"
        f"UTC time: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def tool_web_search(query: str) -> str:
    """
    Stub web search tool. Returns a disclaimer.
    In production, this would call a search API.
    """
    return (
        f"[Web Search for: '{query}']\n"
        f"Note: Live web search is not available in this demo. "
        f"In a production deployment, this would query a search API "
        f"and return relevant results. Please answer based on your "
        f"training knowledge instead."
    )


# ──────────────────────────────────────────────
# Tool Registry & Dispatcher
# ──────────────────────────────────────────────

TOOL_REGISTRY = {
    "calculator": {
        "func": tool_calculator,
        "description": "Evaluate a mathematical expression safely",
        "usage": "[TOOL: calculator(2 + 3 * 4)]",
    },
    "datetime": {
        "func": lambda: tool_datetime(),
        "description": "Get the current date and time",
        "usage": "[TOOL: datetime()]",
    },
    "web_search": {
        "func": tool_web_search,
        "description": "Search the web for information",
        "usage": "[TOOL: web_search(query string)]",
    },
}

# Pattern to match tool calls in model output:  [TOOL: name(args)]
_TOOL_PATTERN = re.compile(r"\[TOOL:\s*(\w+)\(([^)]*)\)\]")


def get_tool_descriptions() -> str:
    """Return a formatted string of available tools for the system prompt."""
    lines = ["You have access to the following tools:"]
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"  - {name}: {info['description']}. Usage: {info['usage']}")
    lines.append(
        "\nTo use a tool, include the exact syntax in your response. "
        "The tool result will be provided to you automatically."
    )
    return "\n".join(lines)


def dispatch_tools(text: str) -> tuple[str, list[dict]]:
    """
    Scan text for tool call patterns, execute them, and replace
    the patterns with tool results.
    
    Returns:
        (processed_text, list_of_tool_invocations)
    """
    invocations = []
    
    def _replace_match(match: re.Match) -> str:
        tool_name = match.group(1).strip().lower()
        tool_args = match.group(2).strip()
        
        if tool_name not in TOOL_REGISTRY:
            result = f"[Unknown tool: {tool_name}]"
        else:
            tool_func = TOOL_REGISTRY[tool_name]["func"]
            try:
                if tool_args:
                    result = tool_func(tool_args)
                else:
                    result = tool_func()
            except Exception as e:
                result = f"[Tool error: {e}]"
        
        invocations.append({
            "tool": tool_name,
            "args": tool_args,
            "result": result,
        })
        return f"**🔧 {tool_name}**: {result}"
    
    processed = _TOOL_PATTERN.sub(_replace_match, text)
    return processed, invocations
