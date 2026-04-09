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
RETRIEVAL_KWARGS = {"top_k": 20}


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
            f"- ALWAYS search KB BEFORE answering or acting. Search MULTIPLE times with "
            f"different, specific keywords. For product recommendations, search for EACH "
            f"candidate product individually to get complete details (fees, eligibility, "
            f"promotions, subscriber benefits).\n"
            f"- Before recommending products, ask the customer about their Rho-Bank "
            f"subscription status and existing accounts — these affect eligibility and pricing.\n"
            f"- When KB results mention a tool name, follow the FULL discovery workflow: "
            f"unlock_discoverable_agent_tool(name) → call_discoverable_agent_tool(name, args). "
            f"For user tools: give_discoverable_user_tool(name) and explain usage.\n"
            f"- For user lookup: try get_user_information_by_name AND get_user_information_by_email. "
            f"If one fails, try the other.\n"
            f"- Authenticate (verify 2 of 4: DOB, email, phone, address) then call "
            f"log_verification BEFORE accessing/modifying account data.\n"
            f"- Complete ALL steps. If a tool call fails, search KB with different terms and retry. "
            f"Never give up or transfer to human without exhausting all options."
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
            # After KB search results, nudge the model about any discoverable tools found
            for tm in message.tool_messages:
                content = getattr(tm, 'content', '') or ''
                tool_names = self._extract_tool_names(content)
                if tool_names:
                    nudge = (
                        f"[Note: KB result mentions discoverable tools: {', '.join(tool_names)}. "
                        f"Use unlock_discoverable_agent_tool(name) then call_discoverable_agent_tool(name, args) for agent tools, "
                        f"or give_discoverable_user_tool(name) for user tools.]"
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
