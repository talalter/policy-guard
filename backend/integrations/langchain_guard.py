"""LangChain callback integration for runtime faithfulness checking.

Drop FaithfulnessGuard into any LangChain agent to verify that the agent's
final response is faithful to what its tools actually returned:

    from langchain.agents import AgentExecutor
    from backend.core import Router
    from backend.integrations import FaithfulnessGuard

    guard = FaithfulnessGuard(router=Router())
    executor = AgentExecutor(agent=..., tools=[...], callbacks=[guard])

The guard accumulates tool outputs during the run, then runs the full
NLI + LLM ensemble against the agent's final response when the chain ends.
If violations are found they are logged as warnings.  Pass
raise_on_violation=True to raise FaithfulnessViolationError instead -
useful in test suites or strict pipelines.
"""

import logging

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError as exc:
    raise ImportError(
        "langchain-core is required for FaithfulnessGuard. "
        "Install it with: pip install langchain-core"
    ) from exc

from backend.core.router import Router
from backend.models import Violation

logger = logging.getLogger(__name__)


class FaithfulnessViolationError(Exception):
    """Raised by FaithfulnessGuard when the agent response violates policy."""

    def __init__(self, violations: list[Violation]) -> None:
        """Store violations and build a human-readable message."""
        self.violations = violations
        count = len(violations)
        super().__init__(
            f"Agent response contains {count} violation(s) with tool outputs."
        )


class FaithfulnessGuard(BaseCallbackHandler):
    """LangChain callback that checks agent responses against tool outputs.

    Accumulates every tool output during a chain run via on_tool_end, then
    runs the full NLI + LLM ensemble against the agent's final response in
    on_chain_end.  Tool outputs are concatenated as the ground-truth context.
    """

    def __init__(
        self,
        router: Router,
        raise_on_violation: bool = False,
    ) -> None:
        """Initialise the guard with a Router (full NLI + LLM ensemble).

        Args:
            router: A Router instance - runs NLI first, escalates uncertain
                pairs to the LLM judge only when needed.
            raise_on_violation: If True, raise FaithfulnessViolationError
                instead of logging a warning when violations are found.
        """
        super().__init__()
        self._router = router
        self._raise_on_violation = raise_on_violation
        self._tool_outputs: list[str] = []

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Accumulate tool output for the faithfulness check."""
        self._tool_outputs.append(output)
        logger.debug("FaithfulnessGuard: collected tool output (%d chars)", len(output))

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        """Check the agent's final response against accumulated tool outputs."""
        if not self._tool_outputs:
            return
        context = "\n\n---\n\n".join(self._tool_outputs)
        response = outputs.get("output") or outputs.get("text") or ""
        if response:
            self._check_and_reset(context, response)

    def _check_and_reset(self, context: str, response: str) -> None:
        """Run the full ensemble check, then clear accumulated tool outputs."""
        try:
            violations, _ = self._router.route(context, response)
            self._report(violations)
        finally:
            self._tool_outputs.clear()

    def _report(self, violations: list[Violation]) -> None:
        """Log or raise findings depending on raise_on_violation."""
        if not violations:
            logger.debug("FaithfulnessGuard: response is faithful to tool outputs")
            return
        logger.warning(
            "FaithfulnessGuard: %d violation(s) detected in agent response",
            len(violations),
        )
        for v in violations:
            logger.warning(
                "  [%s | conf=%.2f] %s", v.severity.value, v.confidence, v.explanation
            )
        if self._raise_on_violation:
            raise FaithfulnessViolationError(violations)
