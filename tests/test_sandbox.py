"""Tests for LocalSandbox and DockerSandbox."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from issue_to_pr.sandbox import (
    DockerSandbox,
    LocalSandbox,
    SandboxError,
    validate_relative_path,
)


def test_validate_relative_path_blocks_traversal():
    assert validate_relative_path("src\\app\\main.py") == "src/app/main.py"
    with pytest.raises(SandboxError):
        validate_relative_path("../secrets.txt")
    with pytest.raises(SandboxError):
        validate_relative_path("/etc/passwd")


def test_local_sandbox_lifecycle_and_file_ops():
    with LocalSandbox() as sb:
        sb.write_file("pkg/mod.py", "VALUE = 42\n")
        assert sb.file_exists("pkg/mod.py")
        assert sb.read_file("pkg/mod.py") == "VALUE = 42\n"
        assert "pkg/mod.py" in sb.list_files()
        res = sb.run([sb.python_executable, "-c", "print('ok')"])
        assert res.exit_code == 0
        assert res.stdout.strip() == "ok"


def test_docker_sandbox_builds_hardened_commands():
    sb = DockerSandbox(image="custom:1.0", network_mode="none")
    start_cmd = sb._build_start_command()
    assert "--memory=1g" in start_cmd
    assert "--cpus=1" in start_cmd
    assert "--pids-limit=256" in start_cmd
    assert "--security-opt=no-new-privileges" in start_cmd
    assert "--network" in start_cmd and "none" in start_cmd

    with patch("issue_to_pr.sandbox.subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="container123\n", stderr=""),
            MagicMock(returncode=0, stdout="hello\n", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]
        with sb:
            res = sb.run("echo hello")
            assert res.exit_code == 0
            assert res.stdout.strip() == "hello"

        assert mock_run.call_count == 3
        rm_call = mock_run.call_args_list[2].args[0]
        assert rm_call == ["docker", "rm", "-f", "container123"]
