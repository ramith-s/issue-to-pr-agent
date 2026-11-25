"""GitHub REST API client and local bare-repo equivalent for offline demos."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import requests

from issue_to_pr.models import CreatedPullRequest, GitHubIssue

_ISSUE_URL_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<number>\d+)/?$"
)
_ISSUE_SHORT_RE = re.compile(
    r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)#(?P<number>\d+)$"
)


class GitHubError(RuntimeError):
    """Raised on GitHub API failures or invalid issue references."""


def parse_issue_reference(issue_ref: str) -> tuple[str, str, int]:
    """Parse `https://github.com/owner/repo/issues/123` or `owner/repo#123`."""
    cleaned = issue_ref.strip()
    for pattern in (_ISSUE_URL_RE, _ISSUE_SHORT_RE):
        match = pattern.match(cleaned)
        if match:
            return match.group("owner"), match.group("repo"), int(match.group("number"))
    raise GitHubError(
        f"Invalid GitHub issue reference {issue_ref!r}. "
        "Expected 'https://github.com/owner/repo/issues/123' or 'owner/repo#123'."
    )


class IssueTrackerClient(Protocol):
    """Common interface implemented by live GitHub and local demo clients."""

    def fetch_issue(self, issue_ref: str) -> GitHubIssue:
        ...

    def get_clone_url(self, issue: GitHubIssue) -> str:
        ...

    def create_pull_request(
        self,
        issue: GitHubIssue,
        title: str,
        body: str,
        head_branch: str,
        dry_run: bool = False,
    ) -> CreatedPullRequest:
        ...


class GitHubClient:
    """Thin GitHub REST API v3 client for issues and Pull Requests."""

    def __init__(
        self,
        token: str = "",
        api_base: str = "https://api.github.com",
        session: requests.Session | None = None,
    ) -> None:
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "issue-to-pr-agent/0.1.0",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.api_base}{path}"
        response = self.session.request(method, url, timeout=30, **kwargs)
        if response.status_code >= 400:
            raise GitHubError(
                f"GitHub API {method} {path} failed ({response.status_code}): {response.text[:400]}"
            )
        return response.json()

    def fetch_issue(self, issue_ref: str) -> GitHubIssue:
        owner, repo, number = parse_issue_reference(issue_ref)
        issue_data = self._request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        repo_data = self._request("GET", f"/repos/{owner}/{repo}")
        labels = [
            lbl["name"]
            for lbl in issue_data.get("labels", [])
            if isinstance(lbl, dict) and "name" in lbl
        ]
        return GitHubIssue(
            owner=owner,
            repo=repo,
            number=number,
            title=issue_data["title"],
            body=issue_data.get("body") or "",
            url=issue_data.get("html_url") or f"https://github.com/{owner}/{repo}/issues/{number}",
            labels=labels,
            default_branch=repo_data.get("default_branch", "main"),
        )

    def get_clone_url(self, issue: GitHubIssue) -> str:
        if self.token:
            safe_token = quote(self.token, safe="")
            return f"https://x-access-token:{safe_token}@github.com/{issue.owner}/{issue.repo}.git"
        return f"https://github.com/{issue.owner}/{issue.repo}.git"

    def create_pull_request(
        self,
        issue: GitHubIssue,
        title: str,
        body: str,
        head_branch: str,
        dry_run: bool = False,
    ) -> CreatedPullRequest:
        if dry_run:
            return CreatedPullRequest(
                number=0,
                url=f"https://github.com/{issue.owner}/{issue.repo}/compare/{issue.default_branch}...{head_branch}",
                title=title,
                body=body,
                head_branch=head_branch,
                base_branch=issue.default_branch,
                dry_run=True,
            )
        if not self.token:
            raise GitHubError("GITHUB_TOKEN is required to open a live Pull Request.")
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": issue.default_branch,
        }
        pr_data = self._request(
            "POST",
            f"/repos/{issue.owner}/{issue.repo}/pulls",
            json=payload,
        )
        return CreatedPullRequest(
            number=pr_data["number"],
            url=pr_data["html_url"],
            title=pr_data["title"],
            body=pr_data.get("body") or body,
            head_branch=head_branch,
            base_branch=issue.default_branch,
            dry_run=False,
        )


class LocalGitHubClient:
    """Offline IssueTrackerClient backed by a local bare git repository and JSON issue."""

    def __init__(
        self,
        bare_repo_path: Path,
        issue: GitHubIssue,
        output_dir: Path | None = None,
    ) -> None:
        self.bare_repo_path = Path(bare_repo_path).resolve()
        self.issue = issue
        self.output_dir = output_dir
        self.created_prs: list[CreatedPullRequest] = []

    def fetch_issue(self, issue_ref: str) -> GitHubIssue:
        return self.issue

    def get_clone_url(self, issue: GitHubIssue) -> str:
        return str(self.bare_repo_path)

    def create_pull_request(
        self,
        issue: GitHubIssue,
        title: str,
        body: str,
        head_branch: str,
        dry_run: bool = False,
    ) -> CreatedPullRequest:
        pr_number = len(self.created_prs) + 1
        pr = CreatedPullRequest(
            number=pr_number,
            url=f"https://github.com/{issue.owner}/{issue.repo}/pull/{pr_number}",
            title=title,
            body=body,
            head_branch=head_branch,
            base_branch=issue.default_branch,
            dry_run=dry_run,
        )
        self.created_prs.append(pr)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_file = self.output_dir / f"pr_{pr_number}.json"
            out_file.write_text(json.dumps(pr.model_dump(), indent=2), encoding="utf-8")
        return pr
