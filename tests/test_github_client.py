"""Tests for GitHub issue parsing and REST API client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from issue_to_pr.github_client import GitHubClient, GitHubError, parse_issue_reference
from issue_to_pr.models import GitHubIssue


def test_parse_issue_reference_variants():
    assert parse_issue_reference("https://github.com/octocat/hello-world/issues/99") == (
        "octocat",
        "hello-world",
        99,
    )
    assert parse_issue_reference("octocat/hello-world#99") == ("octocat", "hello-world", 99)


def test_parse_issue_reference_invalid():
    with pytest.raises(GitHubError, match="Invalid GitHub issue reference"):
        parse_issue_reference("https://example.com/not-github/1")


def test_github_client_fetch_and_create_pr():
    session = MagicMock()
    issue_resp = MagicMock(status_code=200)
    issue_resp.json.return_value = {
        "title": "Fix edge case",
        "body": "Details",
        "html_url": "https://github.com/acme/app/issues/7",
        "labels": [{"name": "bug"}],
    }
    repo_resp = MagicMock(status_code=200)
    repo_resp.json.return_value = {"default_branch": "develop"}
    pr_resp = MagicMock(status_code=201)
    pr_resp.json.return_value = {
        "number": 12,
        "html_url": "https://github.com/acme/app/pull/12",
        "title": "fix: resolve #7",
        "body": "Explanation",
    }
    session.request.side_effect = [issue_resp, repo_resp, pr_resp]

    client = GitHubClient(token="secret-token", session=session)
    issue = client.fetch_issue("acme/app#7")
    assert issue == GitHubIssue(
        owner="acme",
        repo="app",
        number=7,
        title="Fix edge case",
        body="Details",
        url="https://github.com/acme/app/issues/7",
        labels=["bug"],
        default_branch="develop",
    )

    pr = client.create_pull_request(
        issue=issue,
        title="fix: resolve #7",
        body="Explanation",
        head_branch="fix/issue-7-fix-edge-case",
        dry_run=False,
    )
    assert pr.number == 12
    assert pr.base_branch == "develop"
    assert pr.url == "https://github.com/acme/app/pull/12"
