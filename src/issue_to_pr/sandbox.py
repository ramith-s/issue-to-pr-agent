"""Isolated execution sandboxes for cloning, testing, and patching repositories."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a command executed inside a sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        """Return stdout and stderr joined cleanly for diagnostics."""
        parts = [part.strip() for part in (self.stdout, self.stderr) if part.strip()]
        return "\n".join(parts)


class SandboxError(RuntimeError):
    """Raised when sandbox setup or path validation fails."""


def validate_relative_path(rel_path: str) -> str:
    """Validate that a path is strictly relative and cannot escape `/workspace`."""
    normalized = rel_path.replace("\\", "/").strip()
    if not normalized:
        raise SandboxError("Path cannot be empty.")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts:
        raise SandboxError(f"Unsafe path outside workspace: {rel_path!r}")
    return posix.as_posix()


class Sandbox(ABC):
    """Abstract interface for isolated repository execution environments."""

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    @abstractmethod
    def start(self) -> None:
        """Initialize the sandbox environment."""

    @abstractmethod
    def cleanup(self) -> None:
        """Destroy the sandbox and release temporary resources."""

    @abstractmethod
    def run(self, command: str | list[str], timeout: int = 120) -> CommandResult:
        """Run a command inside the workspace root."""

    @abstractmethod
    def read_file(self, rel_path: str) -> str:
        """Read a UTF-8 file relative to the workspace root."""

    @abstractmethod
    def write_file(self, rel_path: str, content: str) -> None:
        """Write a UTF-8 file relative to the workspace root."""

    @abstractmethod
    def file_exists(self, rel_path: str) -> bool:
        """Return True if a file exists at `rel_path` inside the workspace."""

    @abstractmethod
    def list_files(self, ignored_dirs: set[str] | None = None) -> list[str]:
        """List relative POSIX file paths in the workspace."""

    @property
    @abstractmethod
    def python_executable(self) -> str:
        """Python command used to invoke pytest inside this sandbox."""


class LocalSandbox(Sandbox):
    """Temporary-directory sandbox for local development, demos, and unit tests."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.workspace_dir: Path | None = None

    @property
    def python_executable(self) -> str:
        return sys.executable

    def start(self) -> None:
        if self.workspace_dir is not None:
            return
        self._temp_dir = tempfile.TemporaryDirectory(
            prefix="issue2pr_",
            dir=str(self._base_dir) if self._base_dir else None,
        )
        self.workspace_dir = Path(self._temp_dir.name).resolve()

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                shutil.rmtree(self._temp_dir.name, ignore_errors=True)
            self._temp_dir = None
            self.workspace_dir = None

    def _resolve(self, rel_path: str) -> Path:
        if self.workspace_dir is None:
            raise SandboxError("Sandbox has not been started.")
        safe_rel = validate_relative_path(rel_path)
        target = (self.workspace_dir / safe_rel).resolve()
        if target != self.workspace_dir and self.workspace_dir not in target.parents:
            raise SandboxError(f"Path escapes sandbox workspace: {rel_path!r}")
        return target

    def run(self, command: str | list[str], timeout: int = 120) -> CommandResult:
        if self.workspace_dir is None:
            raise SandboxError("Sandbox has not been started.")
        use_shell = isinstance(command, str)
        try:
            proc = subprocess.run(
                command,
                cwd=self.workspace_dir,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or f"Timed out after {timeout}s"),
                timed_out=True,
            )

    def read_file(self, rel_path: str) -> str:
        target = self._resolve(rel_path)
        return target.read_text(encoding="utf-8", errors="replace")

    def write_file(self, rel_path: str, content: str) -> None:
        target = self._resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def file_exists(self, rel_path: str) -> bool:
        return self._resolve(rel_path).is_file()

    def list_files(self, ignored_dirs: set[str] | None = None) -> list[str]:
        if self.workspace_dir is None:
            raise SandboxError("Sandbox has not been started.")
        ignored = ignored_dirs or {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        results: list[str] = []
        for path in sorted(self.workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace_dir)
            if any(part in ignored for part in rel.parts):
                continue
            results.append(rel.as_posix())
        return results


class DockerSandbox(Sandbox):
    """Hardened Docker container sandbox running repository code as non-root."""

    def __init__(
        self,
        image: str = "issue-to-pr-sandbox:latest",
        memory_limit: str = "1g",
        cpu_limit: str = "1",
        pids_limit: int = 256,
        network_mode: str | None = None,
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.network_mode = network_mode
        self.container_id: str | None = None

    @property
    def python_executable(self) -> str:
        return "python"

    def _build_start_command(self) -> list[str]:
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_limit}",
            f"--pids-limit={self.pids_limit}",
            "--security-opt=no-new-privileges",
            "-w",
            "/workspace",
        ]
        if self.network_mode:
            cmd.extend(["--network", self.network_mode])
        cmd.extend([self.image, "sleep", "infinity"])
        return cmd

    def start(self) -> None:
        if self.container_id is not None:
            return
        proc = subprocess.run(
            self._build_start_command(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SandboxError(
                f"Failed to start Docker sandbox ({self.image}): {proc.stderr.strip()}"
            )
        self.container_id = proc.stdout.strip()

    def cleanup(self) -> None:
        if self.container_id is None:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        self.container_id = None

    def run(self, command: str | list[str], timeout: int = 120) -> CommandResult:
        if self.container_id is None:
            raise SandboxError("Docker sandbox has not been started.")
        if isinstance(command, list):
            exec_cmd = ["docker", "exec", self.container_id, *command]
        else:
            exec_cmd = ["docker", "exec", self.container_id, "sh", "-lc", command]
        try:
            proc = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CommandResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or f"Timed out after {timeout}s"),
                timed_out=True,
            )

    def read_file(self, rel_path: str) -> str:
        safe_rel = validate_relative_path(rel_path)
        result = self.run(["cat", f"/workspace/{safe_rel}"])
        if result.exit_code != 0:
            raise FileNotFoundError(f"{safe_rel}: {result.stderr.strip()}")
        return result.stdout

    def write_file(self, rel_path: str, content: str) -> None:
        safe_rel = validate_relative_path(rel_path)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        # Decode base64 inside container so arbitrary quotes/newlines are preserved safely.
        script = (
            f"mkdir -p \"$(dirname '/workspace/{safe_rel}')\" && "
            f"python -c \"import base64, pathlib; "
            f"pathlib.Path('/workspace/{safe_rel}').write_bytes(base64.b64decode('{encoded}'))\""
        )
        result = self.run(script)
        if result.exit_code != 0:
            raise SandboxError(f"Failed writing {safe_rel}: {result.combined_output}")

    def file_exists(self, rel_path: str) -> bool:
        safe_rel = validate_relative_path(rel_path)
        return self.run(["test", "-f", f"/workspace/{safe_rel}"]).exit_code == 0

    def list_files(self, ignored_dirs: set[str] | None = None) -> list[str]:
        ignored = ignored_dirs or {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        prune_expr = " -o ".join(f"-name {name}" for name in sorted(ignored))
        cmd = f"find . \\( {prune_expr} \\) -prune -o -type f -print | sed 's|^\\./||' | sort"
        result = self.run(cmd)
        if result.exit_code != 0:
            raise SandboxError(f"Failed listing workspace files: {result.combined_output}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def create_sandbox(
    backend: Literal["docker", "local"],
    docker_image: str = "issue-to-pr-sandbox:latest",
) -> Sandbox:
    """Instantiate the requested sandbox backend."""
    if backend == "docker":
        return DockerSandbox(image=docker_image)
    if backend == "local":
        return LocalSandbox()
    raise ValueError(f"Unsupported sandbox backend: {backend!r}")
