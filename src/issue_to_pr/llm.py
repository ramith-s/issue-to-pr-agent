"""LLM adapters for repo exploration, failing test generation, fixing, and PR drafting."""

from __future__ import annotations

import json
import re
from typing import Protocol, TypeVar

import requests
from pydantic import BaseModel

from issue_to_pr.models import (
    ExplorationPlan,
    FileEdit,
    GitHubIssue,
    PatchPlan,
    PullRequestDraft,
    ReproTestPlan,
)

TModel = TypeVar("TModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM request or structured JSON parse fails."""


class CodingLLM(Protocol):
    """Protocol implemented by OpenAI, Anthropic, and deterministic demo backends."""

    def plan_exploration(
        self,
        issue: GitHubIssue,
        repo_files: list[str],
        readme_excerpt: str,
    ) -> ExplorationPlan:
        ...

    def generate_repro_test(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        previous_feedback: str | None = None,
    ) -> ReproTestPlan:
        ...

    def generate_fix(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        failing_output: str,
        previous_attempts: list[str],
    ) -> PatchPlan:
        ...

    def draft_pull_request(
        self,
        issue: GitHubIssue,
        git_diff: str,
        test_summary: str,
    ) -> PullRequestDraft:
        ...


def _extract_json_object(raw_text: str) -> dict:
    """Strip optional markdown code fences and parse a JSON object."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model did not return valid JSON: {exc}\nRaw: {raw_text[:400]}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Expected a JSON object from the model.")
    return parsed


def _format_files_for_prompt(file_contents: dict[str, str]) -> str:
    sections = []
    for path, content in file_contents.items():
        sections.append(f"### FILE: {path}\n```python\n{content}\n```")
    return "\n\n".join(sections)


class _StructuredPromptLLM:
    """Shared prompt formatting and structured response parsing across HTTP LLM providers."""

    def _complete_json(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    def _ask_structured(
        self,
        schema: type[TModel],
        system_prompt: str,
        user_prompt: str,
    ) -> TModel:
        raw = self._complete_json(system_prompt, user_prompt)
        data = _extract_json_object(raw)
        return schema.model_validate(data)

    def plan_exploration(
        self,
        issue: GitHubIssue,
        repo_files: list[str],
        readme_excerpt: str,
    ) -> ExplorationPlan:
        system = (
            "You are a senior software engineer investigating a bug report. "
            "Respond ONLY with valid JSON matching keys: summary (str), "
            "relevant_files (list[str]), search_terms (list[str])."
        )
        file_list = "\n".join(repo_files[:200])
        user = (
            f"Issue #{issue.number}: {issue.title}\n\n"
            f"{issue.body}\n\n"
            f"README excerpt:\n{readme_excerpt[:1500]}\n\n"
            f"Repository files:\n{file_list}\n\n"
            "Select up to 8 most relevant source and test files to read next."
        )
        return self._ask_structured(ExplorationPlan, system, user)

    def generate_repro_test(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        previous_feedback: str | None = None,
    ) -> ReproTestPlan:
        system = (
            "You are a software engineer writing a minimal pytest test that REPRODUCES a bug. "
            "The test MUST fail on the current buggy code and pass once fixed. "
            "Respond ONLY with valid JSON matching keys: "
            "test_file_path (str), test_code (str), rationale (str)."
        )
        feedback_block = (
            f"\nPrevious attempt feedback (must fix):\n{previous_feedback}\n"
            if previous_feedback
            else ""
        )
        user = (
            f"Issue #{issue.number}: {issue.title}\n\n"
            f"{issue.body}\n"
            f"{feedback_block}\n"
            f"Relevant files:\n{_format_files_for_prompt(file_contents)}"
        )
        return self._ask_structured(ReproTestPlan, system, user)

    def generate_fix(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        failing_output: str,
        previous_attempts: list[str],
    ) -> PatchPlan:
        system = (
            "You are a senior software engineer fixing a verified bug. "
            "Make the smallest clean change to fix the root cause without breaking existing tests. "
            "Respond ONLY with valid JSON matching keys: diagnosis (str), "
            "edits (list of {path: str, content: str, reason: str}) where content is the FULL updated file."
        )
        history = "\n".join(f"- {note}" for note in previous_attempts) or "None"
        user = (
            f"Issue #{issue.number}: {issue.title}\n\n"
            f"{issue.body}\n\n"
            f"Previous fix attempts:\n{history}\n\n"
            f"Failing pytest output:\n{failing_output[-3500:]}\n\n"
            f"Current files:\n{_format_files_for_prompt(file_contents)}"
        )
        return self._ask_structured(PatchPlan, system, user)

    def draft_pull_request(
        self,
        issue: GitHubIssue,
        git_diff: str,
        test_summary: str,
    ) -> PullRequestDraft:
        system = (
            "You are writing a clear, concise GitHub Pull Request description. "
            "Respond ONLY with valid JSON matching keys: title (str), body (str markdown)."
        )
        user = (
            f"Issue #{issue.number}: {issue.title}\n"
            f"Issue body:\n{issue.body}\n\n"
            f"Pytest summary: {test_summary}\n\n"
            f"Git diff:\n{git_diff[:6000]}"
        )
        return self._ask_structured(PullRequestDraft, system, user)


class OpenAIChatLLM(_StructuredPromptLLM):
    """OpenAI Chat Completions adapter using JSON object mode."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required for the 'openai' LLM provider.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _complete_json(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise LLMError(f"OpenAI API error ({resp.status_code}): {resp.text[:400]}")
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicChatLLM(_StructuredPromptLLM):
    """Anthropic Messages API adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is required for the 'anthropic' LLM provider.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _complete_json(self, system_prompt: str, user_prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "temperature": 0.1,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise LLMError(f"Anthropic API error ({resp.status_code}): {resp.text[:400]}")
        return resp.json()["content"][0]["text"]


class ScriptedDemoLLM:
    """Deterministic LLM backend for the bundled sample repository and E2E tests."""

    def __init__(self, simulate_failed_first_fix: bool = False) -> None:
        self.simulate_failed_first_fix = simulate_failed_first_fix

    def plan_exploration(
        self,
        issue: GitHubIssue,
        repo_files: list[str],
        readme_excerpt: str,
    ) -> ExplorationPlan:
        preferred = [
            f for f in repo_files
            if f.endswith(".py") and ("billing" in f or "invoice" in f or "test_" in f)
        ]
        return ExplorationPlan(
            summary="Identified invoice calculation module and existing pytest suite.",
            relevant_files=preferred or repo_files[:6],
            search_terms=["calculate_invoice", "discount", "tax"],
        )

    def generate_repro_test(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        previous_feedback: str | None = None,
    ) -> ReproTestPlan:
        test_code = (
            '"""Regression test reproducing GitHub Issue '
            f'#{issue.number}."""\n\n'
            "from decimal import Decimal\n"
            "from invoice_engine.billing import LineItem, calculate_invoice_total\n\n\n"
            "def test_flat_discount_exceeding_subtotal_never_produces_negative_tax_or_total():\n"
            '    """Issue #42: flat promo coupons exceeding subtotal must clamp at zero before tax."""\n'
            "    items = [LineItem(sku='PRO-1', unit_price=Decimal('25.00'), quantity=1)]\n"
            "    summary = calculate_invoice_total(\n"
            "        items=items,\n"
            "        tax_rate=Decimal('0.10'),\n"
            "        flat_discount=Decimal('40.00'),\n"
            "    )\n"
            "    assert summary.discounted_subtotal == Decimal('0.00')\n"
            "    assert summary.tax == Decimal('0.00')\n"
            "    assert summary.total == Decimal('0.00')\n"
        )
        return ReproTestPlan(
            test_file_path=f"tests/test_issue_{issue.number}.py",
            test_code=test_code,
            rationale=(
                "Reproduces negative discounted subtotal and negative tax when a flat "
                "promotional discount exceeds the line-item subtotal."
            ),
        )

    def generate_fix(
        self,
        issue: GitHubIssue,
        file_contents: dict[str, str],
        failing_output: str,
        previous_attempts: list[str],
    ) -> PatchPlan:
        target_path = next(
            (p for p in file_contents if p.endswith("billing.py")),
            "src/invoice_engine/billing.py",
        )
        source = file_contents[target_path]

        if self.simulate_failed_first_fix and not previous_attempts:
            # Deliberately introduce an incomplete patch on attempt 1 so tests can verify retries.
            broken = source.replace(
                "discounted_subtotal = subtotal - flat_discount",
                "discounted_subtotal = subtotal - flat_discount  # partial attempt",
            )
            return PatchPlan(
                diagnosis="Attempt 1: testing retry loop before clamping.",
                edits=[
                    FileEdit(
                        path=target_path,
                        content=broken,
                        reason="Incomplete initial attempt to exercise retry handling.",
                    )
                ],
            )

        fixed = source.replace(
            "discounted_subtotal = subtotal - flat_discount",
            "discounted_subtotal = max(Decimal(\"0.00\"), subtotal - flat_discount)",
        )
        if fixed == source:
            raise LLMError(f"Could not locate expected buggy expression in {target_path}.")
        return PatchPlan(
            diagnosis=(
                "`calculate_invoice_total` subtracted `flat_discount` from `subtotal` without "
                "clamping at `Decimal('0.00')`, causing negative taxable subtotals and negative totals."
            ),
            edits=[
                FileEdit(
                    path=target_path,
                    content=fixed,
                    reason="Clamp `discounted_subtotal` at zero before applying sales tax.",
                )
            ],
        )

    def draft_pull_request(
        self,
        issue: GitHubIssue,
        git_diff: str,
        test_summary: str,
    ) -> PullRequestDraft:
        return PullRequestDraft(
            title=f"fix: clamp discounted subtotal at zero (#{issue.number})",
            body=(
                f"Fixes #{issue.number}\n\n"
                "### Root Cause\n"
                "`calculate_invoice_total` subtracted `flat_discount` directly from `subtotal` without "
                "clamping to `Decimal('0.00')`. When a flat promotional coupon exceeded the order subtotal, "
                "the function computed a negative taxable subtotal, negative sales tax, and a negative total.\n\n"
                "### Changes\n"
                "- Added regression test `tests/test_issue_42.py` verifying zero-clamped subtotal, tax, and total.\n"
                "- Updated `src/invoice_engine/billing.py` to clamp `discounted_subtotal` at `Decimal('0.00')` "
                "before computing sales tax.\n\n"
                "### Verification\n"
                f"- `pytest`: `{test_summary}`\n"
            ),
        )


def create_llm(
    provider: str,
    model: str,
    openai_api_key: str = "",
    anthropic_api_key: str = "",
) -> CodingLLM:
    """Factory returning the configured CodingLLM backend."""
    if provider == "openai":
        return OpenAIChatLLM(api_key=openai_api_key, model=model)
    if provider == "anthropic":
        return AnthropicChatLLM(api_key=anthropic_api_key, model=model)
    if provider == "demo":
        return ScriptedDemoLLM()
    raise ValueError(f"Unsupported LLM provider: {provider!r}")
