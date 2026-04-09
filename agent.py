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
RETRIEVAL_VARIANT = "openai_embeddings"
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
            f"{self.domain_policy}\n\n"
            f"## Critical Rules\n"
            f"1. SEARCH FIRST: Before ANY action or answer, search KB with specific queries. "
            f"Search multiple times with varied keywords. Read results carefully for exact tool names and required steps.\n"
            f"2. AUTHENTICATE: Before accessing/modifying customer data, verify identity: "
            f"look up user via get_user_information_by_name (or by_email), ask customer to confirm "
            f"2 of 4 (DOB, email, phone, address), then call log_verification with their info and get_current_time().\n"
            f"3. TOOL DISCOVERY: When KB mentions a tool name (e.g. tool_name_1234):\n"
            f"   - Agent tools: unlock_discoverable_agent_tool(exact_name) THEN call_discoverable_agent_tool(exact_name, args)\n"
            f"   - User tools: give_discoverable_user_tool(exact_name) and explain what args the user needs to provide\n"
            f"4. PERSISTENCE: Complete ALL required steps. Search KB again with different queries if stuck. "
            f"Do not transfer to human or give up until you have exhausted every option in the KB.\n"
            f"5. PRECISION: Use exact tool names from KB. Pass all required arguments. "
            f"When KB says to do something, follow the procedure exactly as written."
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
