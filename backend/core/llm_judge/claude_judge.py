"""Anthropic Claude judge with an agentic verification tool loop."""

import logging

from anthropic import Anthropic

from backend.config import settings
from backend.core.llm_judge.base import (
    BaseLLMJudge,
    _JudgeResponse,
    _MAX_TOOL_ITERATIONS,
    _SYSTEM_PROMPT,
    _VERIFY_SPAN_PARAMS,
    _COMPARE_VALUES_PARAMS,
    _FIND_CONTEXT_PARAMS,
    _REPORT_PARAMS,
    _execute_tool,
)

logger = logging.getLogger(__name__)

_CLAUDE_TOOLS: list[dict] = [
    {"name": "verify_span",
     "description": "Check whether a quoted span appears verbatim (case-insensitive) in the context or response. Call before committing to any finding.",
     "input_schema": _VERIFY_SPAN_PARAMS},
    {"name": "compare_values",
     "description": "Deterministically compare two values for equality. Use for CVSS scores, port numbers, version strings, durations, and rate limits.",
     "input_schema": _COMPARE_VALUES_PARAMS},
    {"name": "find_surrounding_context",
     "description": "Retrieve text surrounding a span to check for negation or conditional scoping by nearby sentences.",
     "input_schema": _FIND_CONTEXT_PARAMS},
    {"name": "report_violations",
     "description": "Submit the final analysis. Call once when all findings are verified.",
     "input_schema": _REPORT_PARAMS},
]


def _process_claude_calls(
    content_blocks,
    context: str,
    response: str,
) -> tuple[_JudgeResponse | None, list[dict]]:
    """Execute Anthropic tool_use blocks; return (report, tool_result_blocks)."""
    tool_results: list[dict] = []
    report: _JudgeResponse | None = None
    for block in content_blocks:
        if block.type != "tool_use":
            continue
        if block.name == "report_violations":
            report = _JudgeResponse.model_validate(block.input)
        else:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _execute_tool(block.name, block.input, context, response),
            })
    return report, tool_results


class ClaudeJudge(BaseLLMJudge):
    """LLM judge backed by Anthropic Claude with an agentic verification tool loop."""

    def __init__(self) -> None:
        """Instantiate the Anthropic client."""
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=claude")
        self._client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        logger.info("ClaudeJudge initialised (model=%s)", settings.claude_model)

    def _call_api(self, context: str, response: str, user_message: str) -> _JudgeResponse:
        """Run the Claude agentic tool loop until report_violations is called."""
        messages: list[dict] = [{"role": "user", "content": user_message}]
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        for i in range(_MAX_TOOL_ITERATIONS):
            force = i == _MAX_TOOL_ITERATIONS - 1
            resp = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=messages, # type: ignore
                tools=_CLAUDE_TOOLS,  # type: ignore
                tool_choice={"type": "tool", "name": "report_violations"} if force else {"type": "auto"},
            )
            # Each request charges for the full growing conversation.
            self._last_input_tokens += resp.usage.input_tokens
            self._last_output_tokens += resp.usage.output_tokens
            messages.append({"role": "assistant", "content": resp.content})
            report, tool_results = _process_claude_calls(resp.content, context, response)
            if report is not None:
                return report
            if not tool_results:
                break
            messages.append({"role": "user", "content": tool_results})
        logger.warning("Claude judge loop exhausted after %d iterations without report_violations call", _MAX_TOOL_ITERATIONS)
        return _JudgeResponse(overall_reasoning="Loop exhausted without report.", violations=[])
