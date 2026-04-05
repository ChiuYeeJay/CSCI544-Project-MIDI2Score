from __future__ import annotations

import subprocess
from pathlib import Path


def collect_git_metadata(repo_root: str | Path = ".") -> dict[str, object]:
    root = Path(repo_root)
    head_commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    branch = _run_git(["branch", "--show-current"], cwd=root)
    status_output = _run_git(["status", "--short"], cwd=root)
    status_lines = [line for line in status_output.splitlines() if line]
    return {
        "repo_root": str(root.resolve()),
        "head_commit": head_commit,
        "branch": branch,
        "is_clean": len(status_lines) == 0,
        "status": status_lines,
    }


def require_clean_git_worktree(repo_root: str | Path = ".") -> dict[str, object]:
    metadata = collect_git_metadata(repo_root)
    if not metadata["is_clean"]:
        raise ValueError(
            "Git worktree is not clean. Commit or stash changes before running a managed "
            "experiment, or explicitly allow a dirty worktree."
        )
    return metadata


def _run_git(args: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
