"""Configuration loading for the Issue-to-PR agent."""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Runtime configuration resolved from environment variables and CLI flags."""

    github_token: str = ""
    llm_provider: Literal["openai", "anthropic", "demo"] = "openai"
    llm_model: str = "gpt-4o"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    sandbox_backend: Literal["docker", "local"] = "local"
    docker_image: str = "issue-to-pr-sandbox:latest"
    max_repro_attempts: int = Field(default=2, ge=1, le=5)
    max_fix_attempts: int = Field(default=4, ge=1, le=10)
    command_timeout_seconds: int = Field(default=120, ge=10, le=600)
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Load configuration from `.env` and process environment variables."""
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        default_model = (
            "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"
        )
        return cls(
            github_token=os.getenv("GITHUB_TOKEN", "").strip(),
            llm_provider=provider,  # type: ignore[arg-type]
            llm_model=os.getenv("LLM_MODEL", default_model).strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            sandbox_backend=os.getenv("SANDBOX_BACKEND", "local").strip().lower(),  # type: ignore[arg-type]
            docker_image=os.getenv("DOCKER_IMAGE", "issue-to-pr-sandbox:latest").strip(),
            max_repro_attempts=int(os.getenv("MAX_REPRO_ATTEMPTS", "2")),
            max_fix_attempts=int(os.getenv("MAX_FIX_ATTEMPTS", "4")),
            command_timeout_seconds=int(os.getenv("COMMAND_TIMEOUT_SECONDS", "120")),
            dry_run=os.getenv("DRY_RUN", "false").strip().lower() in {"1", "true", "yes"},
        )
