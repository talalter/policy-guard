"""Centralised application settings using pydantic-settings.

All configuration is read once at import time from environment variables
(and from a .env file if present).  Every other module imports `settings`
from here instead of calling os.getenv directly.

Validation happens at startup: a missing required field (e.g. OPENAI_API_KEY)
or a wrong type (e.g. NLI_TOP_K=abc) raises a clear ValidationError before
any model weights are loaded.
"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """All runtime configuration — loaded from environment variables / .env file."""

    # LLM provider — "openai" (default) or "claude".
    # Switch by setting LLM_PROVIDER in .env; the corresponding API key must also be set.
    llm_provider: Literal["openai", "claude"] = "openai"

    # OpenAI — required when llm_provider=openai.
    # SecretStr prevents the key from appearing in logs or repr() output.
    openai_api_key: SecretStr

    # Anthropic — required when llm_provider=claude.
    anthropic_api_key: SecretStr | None = None

    # NLI scorer
    nli_model: str = "dleemiller/ModernCE-base-nli"
    bi_encoder_model: str = "BAAI/bge-small-en-v1.5"
    nli_top_k: int = 10
    nli_min_similarity: float = 0.25
    nli_mini_batch_size: int = 8
    nli_max_length: int = 512
    nli_confidence_threshold: float = 0.7
    # Router
    direct_severity_threshold: float = 0.90
    # Absolute floor for LLM escalation — independent of nli_confidence_threshold.
    # Pairs whose contradiction_score exceeds this go to GPT-5.4-mini even when NLI
    # is not confident enough to flag them directly.
    nli_escalation_floor: float = 0.4

    # LLM judge
    gpt_model: str = "gpt-5.4-mini"
    claude_model: str = "claude-opus-4-7"
    llm_min_confidence: float = 0.75
    llm_max_tool_iterations: int = 4
    # Minimum peak NLI contradiction_score required to call the LLM.
    # Ignored when force_llm=True.
    llm_signal_floor: float = 0.20
    force_llm: bool = False

    # MongoDB — optional; omit to disable history persistence
    mongodb_url: str | None = None

    # API server
    frontend_url: str = "http://localhost:5173"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")


settings = Settings() # type: ignore
