"""τ-Knowledge banking agent — the artifact agents evolve.

This file is self-contained: all agent logic lives here. Modify anything.
The agent receives customer messages, domain tools (including KB search),
and must follow the domain policy to resolve banking customer requests.
"""

import json
import os
import re
import sys
from typing import Optional

# Suppress loguru warnings during import so eval.sh can cleanly read RETRIEVAL_VARIANT
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="ERROR")

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate


# ── Configuration ────────────────────────────────────────────────────────────

# Retrieval variant for the banking_knowledge domain.
# Options: "bm25", "openai_embeddings", "qwen_embeddings", "grep_only",
#          "full_kb", "no_knowledge"
# NOTE: "golden_retrieval" is blocked by the eval harness.
RETRIEVAL_VARIANT = "bm25"
RETRIEVAL_KWARGS = {"top_k": 15}


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState:
    """Conversation state container."""

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


# ── Agent Implementation ─────────────────────────────────────────────────────

class BankingAgent(HalfDuplexAgent[AgentState]):
    """Banking customer service agent optimized for tool discovery and procedure following."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-5.4-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {"temperature": 0.0, "seed": 300}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AgentState:
        system_prompt = (
            f"You are an expert Rho-Bank customer service agent.\n\n"
            f"{self.domain_policy}\n\n"
            f"## Strategy\n"
            f"- Search KB MULTIPLE times with different keywords before acting. "
            f"For product recommendations, search for EACH candidate product individually.\n"
            f"- Ask about the customer's Rho-Bank subscription status and existing accounts "
            f"before recommending products — these affect eligibility and pricing.\n"
            f"- For user lookup: try get_user_information_by_name AND get_user_information_by_email. "
            f"If one fails, try the other.\n"
            f"- Before executing a multi-step procedure, plan all required steps. "
            f"Complete ALL steps in order. If a tool call fails, search KB with different terms "
            f"and retry. Never give up or transfer to human without exhausting all options."
        )
        return AgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    @staticmethod
    def _extract_tool_names(text: str) -> list[str]:
        """Extract discoverable tool names (word_word_1234 pattern) from text."""
        if not text:
            return []
        return re.findall(r'\b[a-z][a-z_]+_\d{4}\b', text)

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentState,
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            # After tool results, nudge about ANY discoverable tools found in this response.
            # Always nudge (no dedup) — the model may not have acted on earlier nudges.
            found_tools = set()
            for tm in message.tool_messages:
                content = getattr(tm, 'content', '') or ''
                found_tools.update(self._extract_tool_names(content))
            if found_tools:
                nudge = (
                    f"[Tools found in results: {', '.join(sorted(found_tools))}. "
                    f"To use: unlock_discoverable_agent_tool(name) → call_discoverable_agent_tool(name, args). "
                    f"User tools: give_discoverable_user_tool(name). Act on these NOW.]"
                )
                state.messages.append(
                    SystemMessage(role="system", content=nudge)
                )
        else:
            state.messages.append(message)

        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            **self.llm_args,
        )

        state.messages.append(response)
        return response, state


# ── Factory (required by eval harness) ───────────────────────────────────────

def create_agent(tools, domain_policy, **kwargs):
    """Factory function called by the eval harness."""
    return BankingAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-5.4-mini"),
        llm_args=kwargs.get("llm_args", {"temperature": 0.0, "seed": 300}),
    )
