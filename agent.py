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
            f"{self.domain_policy}\n\n"
            f"## Operating Rules\n"
            f"1. SEARCH KB BEFORE EVERY ACTION: Before answering ANY question or performing ANY action, "
            f"search the knowledge base with specific keywords. Search MULTIPLE times with DIFFERENT "
            f"queries to find all relevant policies, procedures, and tool names. For example, if a "
            f"customer asks about disputing a transaction, search for 'dispute', 'transaction dispute', "
            f"'file dispute credit card', etc.\n\n"
            f"2. FOLLOW KB PROCEDURES EXACTLY: When KB describes a procedure with numbered steps, "
            f"follow EVERY step in order. Do not skip steps. If a step says 'confirm X with customer', "
            f"ask the customer before proceeding. If a step says 'check Y', check Y before moving on.\n\n"
            f"3. IDENTITY VERIFICATION:\n"
            f"   a. Look up the customer: try get_user_information_by_name first, then get_user_information_by_email\n"
            f"   b. Ask customer to provide 2 of 4: date of birth, email, phone, address\n"
            f"   c. COMPARE what customer says against the record. If info doesn't match, verification FAILS\n"
            f"   d. After successful verification: call log_verification with all fields from the user record and get_current_time()\n\n"
            f"4. TOOL DISCOVERY AND USE:\n"
            f"   - When KB mentions a tool name (like tool_name_1234), you MUST use it\n"
            f"   - Agent tools: first unlock_discoverable_agent_tool(exact_name), then call_discoverable_agent_tool(exact_name, args)\n"
            f"   - User tools: call give_discoverable_user_tool(exact_name), then tell customer what arguments to provide\n"
            f"   - If you need a tool but haven't found one, search KB again with different keywords\n\n"
            f"5. PRODUCT RECOMMENDATIONS: Ask about Rho-Bank subscription status and existing accounts "
            f"before recommending products — these affect eligibility, fees, and benefits.\n\n"
            f"6. PERSISTENCE: Complete ALL required actions. If something fails, search KB with different "
            f"terms. Only transfer to human after exhausting every option in the KB."
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
