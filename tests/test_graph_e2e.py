"""End-to-end LangGraph workflow tests using a temporary bare git repository."""

from __future__ import annotations

import json
from pathlib import Path

from issue_to_pr.cli import setup_demo_bare_repo
from issue_to_pr.config import AgentConfig
from issue_to_pr.github_client import LocalGitHubClient
from issue_to_pr.graph import run_issue_to_pr
from issue_to_pr.llm import ScriptedDemoLLM
from issue_to_pr.models import GitHubIssue

ROOT = Path(__file__).resolve().parents[1]


def _load_demo_issue() -> GitHubIssue:
    data = json.loads((ROOT / "examples" / "sample_issue_42.json").read_text(encoding="utf-8"))
    return GitHubIssue.model_validate(data)


def test_end_to_end_repro_fix_and_pr(tmp_path: Path):
    bare_repo = setup_demo_bare_repo(ROOT / "examples" / "sample_buggy_repo", tmp_path)
    issue = _load_demo_issue()
    client = LocalGitHubClient(
        bare_repo_path=bare_repo,
        issue=issue,
        output_dir=tmp_path / "pr_out",
    )
    config = AgentConfig(
        llm_provider="demo",
        sandbox_backend="local",
        max_repro_attempts=2,
        max_fix_attempts=3,
        dry_run=False,
    )

    result = run_issue_to_pr(
        issue_ref="acme-billing/invoice-engine#42",
        github_client=client,
        llm=ScriptedDemoLLM(simulate_failed_first_fix=True),
        config=config,
    )

    assert result["status"] == "succeeded"
    assert result["repro_confirmed"] is True
    assert result["fix_attempt"] == 2
    assert result["last_test_result"].passed is True
    assert "4 passed" in result["last_test_result"].summary

    pr = result["pr_result"]
    assert pr.number == 1
    assert pr.head_branch.startswith("fix/issue-42-")
    assert "Fixes #42" in pr.body
    assert (tmp_path / "pr_out" / "pr_1.json").is_file()
    assert "max(Decimal(\"0.00\"), subtotal - flat_discount)" in result["git_diff"]
