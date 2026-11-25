"""Domain models and LangGraph state schema for the Issue-to-PR workflow."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class GitHubIssue(BaseModel):
    """Normalized representation of a GitHub issue."""

    owner: str
    repo: str
    number: int
    title: str
    body: str = ""
    url: str = ""
    labels: list[str] = Field(default_factory=list)
    default_branch: str = "main"


class FileEdit(BaseModel):
    """Full-file content update produced by the coding LLM."""

    path: str
    content: str
    reason: str = ""


class ExplorationPlan(BaseModel):
    """LLM plan identifying files most relevant to reproducing and fixing the bug."""

    summary: str
    relevant_files: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class ReproTestPlan(BaseModel):
    """LLM-generated pytest file designed to fail on the current buggy code."""

    test_file_path: str
    test_code: str
    rationale: str


class PatchPlan(BaseModel):
    """LLM-generated source edits to resolve the bug."""

    diagnosis: str
    edits: list[FileEdit] = Field(default_factory=list)


class PullRequestDraft(BaseModel):
    """Structured PR title and markdown explanation."""

    title: str
    body: str


class CreatedPullRequest(BaseModel):
    """Result returned after creating (or previewing) a Pull Request."""

    number: int
    url: str
    title: str
    body: str
    head_branch: str
    base_branch: str
    dry_run: bool = False


class PytestResult(BaseModel):
    """Structured result of running pytest inside the sandbox."""

    passed: bool
    exit_code: int
    summary: str
    output: str


class AgentState(TypedDict, total=False):
    """Shared LangGraph state passed between workflow nodes."""

    issue_ref: str
    issue: GitHubIssue
    branch_name: str
    repo_files: list[str]
    relevant_files: list[str]
    exploration_summary: str
    repro_test_path: str
    repro_attempt: int
    repro_confirmed: bool
    repro_output: str
    fix_attempt: int
    previous_patch_notes: list[str]
    last_test_result: PytestResult
    git_diff: str
    pr_result: CreatedPullRequest
    status: Literal["running", "succeeded", "failed"]
    error: str
    events: list[str]
