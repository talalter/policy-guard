"""OpenAI GPT-5.4-mini judge with an agentic verification tool loop."""

import json
import logging

from openai import OpenAI

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

_OPENAI_TOOLS: list[dict] = [
    {"type": "function", "function": {"name": "verify_span",
        "description": "Check whether a quoted span appears verbatim (case-insensitive) in the context or response. Call before committing to any finding.",
        "parameters": _VERIFY_SPAN_PARAMS}},
    {"type": "function", "function": {"name": "compare_values",
        "description": "Deterministically compare two values for equality. Use for CVSS scores, port numbers, version strings, durations, and rate limits.",
        "parameters": _COMPARE_VALUES_PARAMS}},
    {"type": "function", "function": {"name": "find_surrounding_context",
        "description": "Retrieve text surrounding a span to check for negation or conditional scoping by nearby sentences.",
        "parameters": _FIND_CONTEXT_PARAMS}},
    {"type": "function", "function": {"name": "report_violations",
        "description": "Submit the final analysis. Call once when all findings are verified.",
        "parameters": _REPORT_PARAMS}},
]

_OPENAI_FORCE_REPORT: dict = {"type": "function", "function": {"name": "report_violations"}}


def _process_openai_calls(
    tool_calls,
    context: str,
    response: str,
) -> tuple[_JudgeResponse | None, list[dict]]:
    """Execute OpenAI tool calls; return (report, tool_result_messages)."""
    tool_msgs: list[dict] = []
    report: _JudgeResponse | None = None
    for tc in tool_calls or []:
        args = json.loads(tc.function.arguments)
        if tc.function.name == "report_violations":
            report = _JudgeResponse.model_validate(args)
        else:
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _execute_tool(tc.function.name, args, context, response),
            })
    return report, tool_msgs


class OpenAIJudge(BaseLLMJudge):
    """LLM judge backed by OpenAI GPT-5.4-mini with an agentic verification tool loop."""

    def __init__(self) -> None:
        """Instantiate the OpenAI client."""
        self._client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        logger.info("OpenAIJudge initialised (model=%s)", settings.gpt_model)

    def _call_api(self, context: str, response: str, user_message: str) -> _JudgeResponse:
        """Run the OpenAI agentic tool loop until report_violations is called."""
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        extra: dict = {}
        if settings.gpt_model.startswith("o"):
            extra["reasoning_effort"] = "high"
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        for i in range(_MAX_TOOL_ITERATIONS):
            force = i == _MAX_TOOL_ITERATIONS - 1
            resp = self._client.chat.completions.create(
                model=settings.gpt_model,
                messages=messages,  # type: ignore
                tools=_OPENAI_TOOLS,  # type: ignore
                tool_choice=_OPENAI_FORCE_REPORT if force else "auto",  # type: ignore
                **extra,
            )
            if resp.usage:
                # Each request is charged for the full growing conversation, so
                # summing prompt_tokens across iterations gives the true total cost.
                self._last_input_tokens += resp.usage.prompt_tokens
                self._last_output_tokens += resp.usage.completion_tokens
            choice = resp.choices[0]
            messages.append(choice.message) # type: ignore
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                break
            report, tool_msgs = _process_openai_calls(tool_calls, context, response)
            if report is not None:
                return report
            messages.extend(tool_msgs)
        logger.warning("OpenAI judge loop exhausted after %d iterations without report_violations call", _MAX_TOOL_ITERATIONS)
        return _JudgeResponse(overall_reasoning="Loop exhausted without report.", violations=[])
