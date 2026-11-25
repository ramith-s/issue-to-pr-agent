"""LangGraph workflow orchestrating Issue -> Repro Test -> Fix -> Pull Request."""

from __future__ import annotations

import re
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from issue_to_pr.config import AgentConfig
from issue_to_pr.github_client import IssueTrackerClient
from issue_to_pr.llm import CodingLLM
from issue_to_pr.models import AgentState, GitHubIssue
from issue_to_pr.sandbox import Sandbox, create_sandbox
from issue_to_pr.tools import SandboxWorkspace

EventCallback = Callable[[str], None]


def _slugify(text: str, max_len: int = 36) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:max_len].strip("-")) or "bugfix"


def _record(
    state: AgentState,
    message: str,
    on_event: EventCallback | None,
) -> list[str]:
    events = list(state.get("events") or [])
    events.append(message)
    if on_event is not None:
        on_event(message)
    return events


def build_issue_to_pr_graph(
    workspace: SandboxWorkspace,
    github_client: IssueTrackerClient,
    llm: CodingLLM,
    config: AgentConfig,
    on_event: EventCallback | None = None,
):
    """Compile the LangGraph StateGraph for the TDD Issue-to-PR loop."""

    def fetch_and_clone(state: AgentState) -> AgentState:
        issue_ref = state["issue_ref"]
        issue = github_client.fetch_issue(issue_ref)
        branch_name = f"fix/issue-{issue.number}-{_slugify(issue.title)}"
        events = _record(
            state,
            f"[fetch] Loaded issue #{issue.number}: {issue.title}",
            on_event,
        )
        clone_url = github_client.get_clone_url(issue)
        workspace.clone_and_branch(
            clone_url=clone_url,
            base_branch=issue.default_branch,
            feature_branch=branch_name,
        )
        repo_files = workspace.list_repo_files()
        events = _record(
            {"events": events},
            f"[clone] Cloned {issue.owner}/{issue.repo} on branch `{branch_name}` ({len(repo_files)} files)",
            on_event,
        )
        return {
            "issue": issue,
            "branch_name": branch_name,
            "repo_files": repo_files,
            "repro_attempt": 0,
            "repro_confirmed": False,
            "fix_attempt": 0,
            "previous_patch_notes": [],
            "status": "running",
            "events": events,
        }

    def explore_repo(state: AgentState) -> AgentState:
        issue: GitHubIssue = state["issue"]
        repo_files = state["repo_files"]
        readme_map = workspace.read_files(["README.md", "readme.md", "pyproject.toml"])
        readme_excerpt = "\n\n".join(f"# {k}\n{v}" for k, v in readme_map.items())

        plan = llm.plan_exploration(issue, repo_files, readme_excerpt)
        selected = [path for path in plan.relevant_files if path in repo_files]

        # Supplement LLM-selected files with keyword hits across the repository.
        search_hits = workspace.search_code(plan.search_terms)
        for hit in search_hits:
            hit_path = hit.split(":", 1)[0]
            if hit_path not in selected and len(selected) < 10:
                selected.append(hit_path)

        if not selected:
            selected = [f for f in repo_files if f.endswith(".py")][:8]

        events = _record(
            state,
            f"[explore] Selected {len(selected)} relevant files: {', '.join(selected[:5])}",
            on_event,
        )
        return {
            "relevant_files": selected,
            "exploration_summary": plan.summary,
            "events": events,
        }

    def write_repro_test(state: AgentState) -> AgentState:
        issue: GitHubIssue = state["issue"]
        attempt = state.get("repro_attempt", 0) + 1
        relevant_contents = workspace.read_files(state["relevant_files"])
        feedback = state.get("repro_output") if attempt > 1 else None

        repro_plan = llm.generate_repro_test(
            issue=issue,
            file_contents=relevant_contents,
            previous_feedback=feedback,
        )
        workspace.write_file(repro_plan.test_file_path, repro_plan.test_code)
        result = workspace.run_pytest(target=repro_plan.test_file_path)

        # Exit code 1 means pytest collected and ran the test, and at least one assertion failed.
        # Exit code 2/3/4 indicates collection/syntax/usage errors, not a valid bug reproduction.
        if not result.passed and result.exit_code == 1:
            confirmed = True
            msg = f"[repro #{attempt}] Confirmed failing test `{repro_plan.test_file_path}` ({result.summary})"
            repro_feedback = result.output
        elif result.passed:
            confirmed = False
            msg = f"[repro #{attempt}] Test unexpectedly passed on buggy code ({result.summary})"
            repro_feedback = (
                "Your test passed on the unmodified code. "
                "Update assertions to trigger the reported bug.\n" + result.output
            )
        else:
            confirmed = False
            msg = f"[repro #{attempt}] Test failed to collect/run cleanly (exit {result.exit_code})"
            repro_feedback = (
                f"pytest exited with code {result.exit_code} (collection/syntax error). "
                f"Fix imports or syntax:\n{result.output}"
            )

        events = _record(state, msg, on_event)
        relevant = list(state["relevant_files"])
        if repro_plan.test_file_path not in relevant:
            relevant.append(repro_plan.test_file_path)

        return {
            "relevant_files": relevant,
            "repro_test_path": repro_plan.test_file_path,
            "repro_attempt": attempt,
            "repro_confirmed": confirmed,
            "repro_output": repro_feedback,
            "last_test_result": result,
            "events": events,
        }

    def implement_fix(state: AgentState) -> AgentState:
        issue: GitHubIssue = state["issue"]
        attempt = state.get("fix_attempt", 0) + 1
        file_contents = workspace.read_files(state["relevant_files"])
        last_result = state["last_test_result"]
        prev_notes = list(state.get("previous_patch_notes") or [])

        patch_plan = llm.generate_fix(
            issue=issue,
            file_contents=file_contents,
            failing_output=last_result.output,
            previous_attempts=prev_notes,
        )
        for edit in patch_plan.edits:
            workspace.write_file(edit.path, edit.content)

        full_result = workspace.run_pytest()
        status_word = "PASSED" if full_result.passed else "FAILED"
        msg = (
            f"[fix #{attempt}] Applied {len(patch_plan.edits)} edit(s); "
            f"full pytest {status_word} ({full_result.summary})"
        )
        events = _record(state, msg, on_event)
        prev_notes.append(
            f"Attempt {attempt} ({patch_plan.diagnosis}) -> {status_word}: {full_result.summary}"
        )
        return {
            "fix_attempt": attempt,
            "previous_patch_notes": prev_notes,
            "last_test_result": full_result,
            "events": events,
        }

    def open_pr(state: AgentState) -> AgentState:
        issue: GitHubIssue = state["issue"]
        branch_name = state["branch_name"]
        test_summary = state["last_test_result"].summary
        diff = workspace.git_diff()

        pr_draft = llm.draft_pull_request(
            issue=issue,
            git_diff=diff,
            test_summary=test_summary,
        )
        workspace.commit_and_push(
            commit_message=f"fix: resolve #{issue.number} - {issue.title}",
            branch_name=branch_name,
            push=not config.dry_run,
        )
        pr_result = github_client.create_pull_request(
            issue=issue,
            title=pr_draft.title,
            body=pr_draft.body,
            head_branch=branch_name,
            dry_run=config.dry_run,
        )
        mode_tag = "DRY-RUN PR" if pr_result.dry_run else "Opened PR"
        events = _record(
            state,
            f"[pr] {mode_tag} #{pr_result.number}: {pr_result.url}",
            on_event,
        )
        return {
            "git_diff": diff,
            "pr_result": pr_result,
            "status": "succeeded",
            "events": events,
        }

    def fail_workflow(state: AgentState) -> AgentState:
        if not state.get("repro_confirmed"):
            reason = (
                f"Failed to reproduce issue after {state.get('repro_attempt', 0)} attempt(s). "
                f"Last output: {state.get('repro_output', '')[:500]}"
            )
        else:
            last = state.get("last_test_result")
            reason = (
                f"Tests still failing after {state.get('fix_attempt', 0)} fix attempt(s). "
                f"Last summary: {last.summary if last else 'unknown'}"
            )
        events = _record(state, f"[failed] {reason}", on_event)
        return {
            "status": "failed",
            "error": reason,
            "events": events,
        }

    def route_after_repro(state: AgentState) -> str:
        if state.get("repro_confirmed"):
            return "implement_fix"
        if state.get("repro_attempt", 0) < config.max_repro_attempts:
            return "write_repro_test"
        return "fail_workflow"

    def route_after_fix(state: AgentState) -> str:
        last_result = state.get("last_test_result")
        if last_result and last_result.passed:
            return "open_pr"
        if state.get("fix_attempt", 0) < config.max_fix_attempts:
            return "implement_fix"
        return "fail_workflow"

    graph = StateGraph(AgentState)
    graph.add_node("fetch_and_clone", fetch_and_clone)
    graph.add_node("explore_repo", explore_repo)
    graph.add_node("write_repro_test", write_repro_test)
    graph.add_node("implement_fix", implement_fix)
    graph.add_node("open_pr", open_pr)
    graph.add_node("fail_workflow", fail_workflow)

    graph.add_edge(START, "fetch_and_clone")
    graph.add_edge("fetch_and_clone", "explore_repo")
    graph.add_edge("explore_repo", "write_repro_test")
    graph.add_conditional_edges(
        "write_repro_test",
        route_after_repro,
        {
            "implement_fix": "implement_fix",
            "write_repro_test": "write_repro_test",
            "fail_workflow": "fail_workflow",
        },
    )
    graph.add_conditional_edges(
        "implement_fix",
        route_after_fix,
        {
            "open_pr": "open_pr",
            "implement_fix": "implement_fix",
            "fail_workflow": "fail_workflow",
        },
    )
    graph.add_edge("open_pr", END)
    graph.add_edge("fail_workflow", END)

    return graph.compile()


def run_issue_to_pr(
    issue_ref: str,
    github_client: IssueTrackerClient,
    llm: CodingLLM,
    config: AgentConfig,
    sandbox: Sandbox | None = None,
    on_event: EventCallback | None = None,
) -> AgentState:
    """Run the full Issue-to-PR workflow inside an isolated sandbox."""
    managed_sandbox = sandbox or create_sandbox(
        backend=config.sandbox_backend,
        docker_image=config.docker_image,
    )
    with managed_sandbox as active_sandbox:
        workspace = SandboxWorkspace(
            sandbox=active_sandbox,
            command_timeout=config.command_timeout_seconds,
        )
        app = build_issue_to_pr_graph(
            workspace=workspace,
            github_client=github_client,
            llm=llm,
            config=config,
            on_event=on_event,
        )
        return app.invoke({"issue_ref": issue_ref, "events": []})
