"""τ-Knowledge banking agent — the artifact agents evolve.

This file is self-contained: all agent logic lives here. Modify anything.
The agent receives customer messages, domain tools (including KB search),
and must follow the domain policy to resolve banking customer requests.
"""

import json
import os
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
    """Baseline banking customer service agent.

    Modify this class to improve performance on the banking_knowledge
    benchmark. The domain_policy already contains authentication rules,
    discoverable tool workflows, and KB search guidance.
    """

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
            f"## Workflow\n"
            f"For every customer interaction, follow these steps IN ORDER:\n\n"
            f"1. LISTEN & SEARCH: Understand the request, then search KB with specific keywords. "
            f"Search MULTIPLE times with different terms to find all relevant policies and tools.\n\n"
            f"2. AUTHENTICATE: Look up user via get_user_information_by_name (also try by_email). "
            f"Ask customer for 2 of 4 (DOB, email, phone, address). Match against the record. "
            f"Call log_verification with get_current_time().\n\n"
            f"3. GET FULL ACCOUNT PICTURE: After auth, search KB for how to view all customer "
            f"accounts and transactions. The KB has discoverable tools for viewing bank accounts, "
            f"checking/savings transactions, debit cards, etc. Search for 'user accounts', "
            f"'bank account transactions', 'debit card' as needed.\n\n"
            f"4. DISCOVER & USE TOOLS: KB results mention tool names like tool_name_1234. "
            f"For each: unlock_discoverable_agent_tool(exact_name) THEN "
            f"call_discoverable_agent_tool(exact_name, args). "
            f"For user tools: give_discoverable_user_tool(exact_name) and explain usage.\n\n"
            f"5. COMPLETE ALL STEPS: Follow KB procedures exactly. Many tasks require MULTIPLE "
            f"discoverable tools (e.g. view accounts AND file dispute AND order replacement). "
            f"Search KB again if you need more tools. Ask about subscription status for product recs.\n\n"
            f"6. Never stop early. If stuck, search KB with different terms. "
            f"Transfer to human only as absolute last resort."
        )
        return AgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: AgentState,
    ) -> tuple[AssistantMessage, AgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
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
