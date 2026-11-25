"""Rich CLI entrypoint for running the Issue-to-PR agent."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from issue_to_pr.config import AgentConfig
from issue_to_pr.github_client import GitHubClient, LocalGitHubClient
from issue_to_pr.graph import run_issue_to_pr
from issue_to_pr.llm import ScriptedDemoLLM, create_llm
from issue_to_pr.models import GitHubIssue


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_demo_bare_repo(source_dir: Path, temp_root: Path) -> Path:
    """Create a temporary bare git remote seeded from `examples/sample_buggy_repo`."""
    work_seed = temp_root / "seed_repo"
    bare_remote = temp_root / "origin.git"
    shutil.copytree(source_dir, work_seed)

    commands = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", "Demo Maintainer"],
        ["git", "config", "user.email", "maintainer@example.com"],
        ["git", "add", "-A"],
        ["git", "commit", "-m", "Initial release with existing billing tests"],
    ]
    for cmd in commands:
        subprocess.run(cmd, cwd=work_seed, check=True, capture_output=True, text=True)

    subprocess.run(
        ["git", "clone", "--bare", str(work_seed), str(bare_remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    return bare_remote


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="issue-to-pr",
        description="Autonomous GitHub Issue-to-PR agent (LangGraph + Sandbox + pytest + GitHub API).",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--issue",
        help="GitHub issue URL (https://github.com/owner/repo/issues/123) or owner/repo#123.",
    )
    target_group.add_argument(
        "--demo",
        action="store_true",
        help="Run a full offline end-to-end demo against examples/sample_buggy_repo.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "demo"],
        help="Override LLM_PROVIDER from environment.",
    )
    parser.add_argument("--model", help="Override LLM_MODEL from environment.")
    parser.add_argument(
        "--sandbox",
        choices=["docker", "local"],
        help="Override SANDBOX_BACKEND from environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip pushing branch and opening the live PR; print PR preview instead.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    config = AgentConfig.from_env()

    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        config.llm_model = args.model
    if args.sandbox:
        config.sandbox_backend = args.sandbox
    if args.dry_run:
        config.dry_run = True

    def on_event(message: str) -> None:
        console.print(f"  [cyan]•[/cyan] {message}")

    if args.demo:
        root = _project_root()
        sample_repo = root / "examples" / "sample_buggy_repo"
        issue_payload = json.loads(
            (root / "examples" / "sample_issue_42.json").read_text(encoding="utf-8")
        )
        issue = GitHubIssue.model_validate(issue_payload)
        output_dir = root / ".demo_output"

        console.print(
            Panel.fit(
                f"[bold green]Issue-to-PR Agent (Offline Demo)[/bold green]\n"
                f"Issue #{issue.number}: {issue.title}\n"
                f"Sandbox: {config.sandbox_backend} | LLM: demo",
                border_style="green",
            )
        )
        with tempfile.TemporaryDirectory(prefix="issue2pr_demo_remote_") as tmp_dir:
            bare_remote = setup_demo_bare_repo(sample_repo, Path(tmp_dir))
            client = LocalGitHubClient(
                bare_repo_path=bare_remote,
                issue=issue,
                output_dir=output_dir,
            )
            llm = ScriptedDemoLLM()
            result = run_issue_to_pr(
                issue_ref=f"{issue.owner}/{issue.repo}#{issue.number}",
                github_client=client,
                llm=llm,
                config=config,
                on_event=on_event,
            )
    else:
        console.print(
            Panel.fit(
                f"[bold blue]Issue-to-PR Agent[/bold blue]\n"
                f"Target: {args.issue}\n"
                f"Sandbox: {config.sandbox_backend} | LLM: {config.llm_provider}/{config.llm_model}",
                border_style="blue",
            )
        )
        client = GitHubClient(token=config.github_token)
        llm = create_llm(
            provider=config.llm_provider,
            model=config.llm_model,
            openai_api_key=config.openai_api_key,
            anthropic_api_key=config.anthropic_api_key,
        )
        result = run_issue_to_pr(
            issue_ref=args.issue,
            github_client=client,
            llm=llm,
            config=config,
            on_event=on_event,
        )

    if result.get("status") != "succeeded":
        console.print(
            Panel(
                result.get("error", "Unknown failure"),
                title="[bold red]Workflow Failed[/bold red]",
                border_style="red",
            )
        )
        return 1

    pr = result["pr_result"]
    console.print(
        Panel(
            f"[bold]Title:[/bold] {pr.title}\n"
            f"[bold]Branch:[/bold] {pr.head_branch} -> {pr.base_branch}\n"
            f"[bold]URL:[/bold] {pr.url}\n\n"
            f"{pr.body}",
            title="[bold green]Pull Request Ready[/bold green]",
            border_style="green",
        )
    )
    diff_text = result.get("git_diff", "")
    if diff_text:
        console.print(Panel(Syntax(diff_text, "diff", theme="monokai"), title="Generated Patch"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
