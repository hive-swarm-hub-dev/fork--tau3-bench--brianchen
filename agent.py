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
            f"## Critical Operating Procedure\n"
            f"1. SEARCH KB FIRST: Before any action or answer, search KB with specific keywords. "
            f"Search multiple times with different terms. Read results CAREFULLY — follow every step "
            f"in the procedure exactly as written. Do not skip steps or take shortcuts.\n"
            f"2. VERIFY IDENTITY: Look up user via get_user_information_by_name (try by_email too if needed). "
            f"Compare customer-provided info against the record. Customer must correctly match 2 of 4 "
            f"(DOB, email, phone, address). If info doesn't match the record, do NOT proceed — "
            f"the customer failed verification. Then call log_verification with get_current_time().\n"
            f"3. CONFIRM BEFORE ACTING: Before executing irreversible actions (filing disputes, "
            f"ordering replacements, closing accounts), confirm ALL details with the customer first "
            f"(shipping address, amounts, specific transactions, etc.). Follow KB procedures step-by-step.\n"
            f"4. TOOL DISCOVERY: When KB mentions a tool name:\n"
            f"   - Agent tools: unlock_discoverable_agent_tool(exact_name) THEN call_discoverable_agent_tool(exact_name, args)\n"
            f"   - User tools: give_discoverable_user_tool(exact_name) and tell the customer what args to provide\n"
            f"5. COMPLETE ALL STEPS: Execute every action the KB procedure requires. Search KB again "
            f"with different terms if stuck. Do not give up or transfer to human without exhausting all options."
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
