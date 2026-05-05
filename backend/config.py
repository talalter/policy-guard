"""Centralised application settings using pydantic-settings.

All configuration is read once at import time from environment variables
(and from a .env file if present).  Every other module imports `settings`
from here instead of calling os.getenv directly.

Validation happens at startup: a missing required field (e.g. OPENAI_API_KEY)
or a wrong type (e.g. NLI_TOP_K=abc) raises a clear ValidationError before
any model weights are loaded.
"""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    """All runtime configuration — loaded from environment variables / .env file."""

    # OpenAI — declared here so pydantic-settings loads it from .env and
    # validates its presence at startup with a clear error message.
    # SecretStr prevents the key from appearing in logs or repr() output.
    openai_api_key: SecretStr

    # NLI scorer
    nli_model: str = "dleemiller/ModernCE-base-nli"
    bi_encoder_model: str = "all-MiniLM-L6-v2"
    nli_top_k: int = 5
    nli_min_similarity: float = 0.25
    nli_mini_batch_size: int = 8
    nli_max_length: int = 8192
    nli_confidence_threshold: float = 0.75

    # Router
    direct_severity_threshold: float = 0.90
    # Absolute floor for LLM escalation — independent of nli_confidence_threshold.
    # Pairs whose contradiction_score exceeds this go to GPT-4o even when NLI
    # is not confident enough to flag them directly.
    nli_escalation_floor: float = 0.4

    # LLM judge
    gpt_model: str = "gpt-4.1-mini"
    llm_min_confidence: float = 0.85

    # API server
    frontend_url: str = "http://localhost:5173"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")


settings = Settings() # type: ignore
