"""High-level repository tools executed safely inside a Sandbox."""

from __future__ import annotations

import re

from issue_to_pr.models import PytestResult
from issue_to_pr.sandbox import Sandbox, SandboxError


class SandboxWorkspace:
    """Typed helper methods for inspecting, testing, and committing inside a Sandbox."""

    def __init__(self, sandbox: Sandbox, command_timeout: int = 120) -> None:
        self.sandbox = sandbox
        self.command_timeout = command_timeout

    def clone_and_branch(
        self,
        clone_url: str,
        base_branch: str,
        feature_branch: str,
    ) -> None:
        """Clone the repository into the sandbox root and create a clean feature branch."""
        clone_res = self.sandbox.run(
            ["git", "clone", "--branch", base_branch, clone_url, "."],
            timeout=self.command_timeout,
        )
        if clone_res.exit_code != 0:
            raise SandboxError(f"git clone failed: {clone_res.combined_output}")

        # Configure local git identity inside the cloned repo only (never global config).
        for key, value in (
            ("user.name", "Issue-to-PR Agent"),
            ("user.email", "issue-to-pr-agent@users.noreply.github.com"),
        ):
            cfg_res = self.sandbox.run(["git", "config", key, value])
            if cfg_res.exit_code != 0:
                raise SandboxError(f"git config {key} failed: {cfg_res.combined_output}")

        branch_res = self.sandbox.run(["git", "checkout", "-b", feature_branch])
        if branch_res.exit_code != 0:
            raise SandboxError(f"git checkout -b {feature_branch} failed: {branch_res.combined_output}")

    def list_repo_files(self) -> list[str]:
        """List non-ignored repository files."""
        return self.sandbox.list_files()

    def read_files(self, paths: list[str], max_chars_per_file: int = 12000) -> dict[str, str]:
        """Read existing files into a `{path: content}` map."""
        contents: dict[str, str] = {}
        for rel_path in paths:
            if not self.sandbox.file_exists(rel_path):
                continue
            text = self.sandbox.read_file(rel_path)
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n... [truncated]"
            contents[rel_path] = text
        return contents

    def search_code(self, terms: list[str], max_hits: int = 40) -> list[str]:
        """Search repository text files for matching symbols or terms."""
        if not terms:
            return []
        pattern = re.compile("|".join(re.escape(t) for t in terms if t), re.IGNORECASE)
        hits: list[str] = []
        for rel_path in self.list_repo_files():
            if not rel_path.endswith((".py", ".md", ".toml", ".yaml", ".yml", ".json")):
                continue
            text = self.sandbox.read_file(rel_path)
            for idx, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{rel_path}:{idx}: {line.strip()}")
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def write_file(self, rel_path: str, content: str) -> None:
        """Write or overwrite a file inside the sandbox."""
        self.sandbox.write_file(rel_path, content)

    def run_pytest(self, target: str | None = None) -> PytestResult:
        """Execute pytest inside the sandbox and parse pass/fail status + summary."""
        cmd = [
            self.sandbox.python_executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--tb=short",
        ]
        if target:
            cmd.append(target)
        res = self.sandbox.run(cmd, timeout=self.command_timeout)
        output = res.combined_output
        summary_lines = [line.strip("= ").strip() for line in output.splitlines() if line.strip("= ").strip()]
        summary = next(
            (
                line
                for line in reversed(summary_lines)
                if re.search(r"\b\d+\s+(passed|failed|error|errors|skipped)\b", line)
            ),
            summary_lines[-1] if summary_lines else f"exit code {res.exit_code}",
        )
        return PytestResult(
            passed=(res.exit_code == 0 and not res.timed_out),
            exit_code=res.exit_code,
            summary=summary,
            output=output,
        )

    def git_diff(self) -> str:
        """Return full working-tree + staged diff including untracked files."""
        self.sandbox.run(["git", "add", "-N", "."])
        res = self.sandbox.run(["git", "diff"])
        return res.stdout

    def commit_and_push(
        self,
        commit_message: str,
        branch_name: str,
        push: bool = True,
    ) -> None:
        """Stage all changes, create a commit, and optionally push the branch to origin."""
        add_res = self.sandbox.run(["git", "add", "-A"])
        if add_res.exit_code != 0:
            raise SandboxError(f"git add failed: {add_res.combined_output}")

        commit_res = self.sandbox.run(["git", "commit", "-m", commit_message])
        if commit_res.exit_code != 0:
            raise SandboxError(f"git commit failed: {commit_res.combined_output}")

        if push:
            push_res = self.sandbox.run(
                ["git", "push", "-u", "origin", branch_name],
                timeout=self.command_timeout,
            )
            if push_res.exit_code != 0:
                safe_out = push_res.combined_output
                # Guard against leaking embedded HTTPS access tokens in error messages.
                safe_out = re.sub(r"x-access-token:[^@]+@", "x-access-token:***@ ", safe_out)
                raise SandboxError(f"git push failed: {safe_out}")
