"""A read-only look at the previous session's copied-out worktree, for a self-contained brief.

It stands for a person opening the repository to find something to link: the file tree,
the commit log and the top of the documents. It never reads transcripts: a tree that
holds session logs (`*.jsonl`, `.claude/projects`) is refused rather than filtered, and
`.claude/`, `.codex/` and dependency directories are not listed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", ".claude", ".codex", ".thalamus", "node_modules", ".venv", "venv",
             "__pycache__", ".cache", "dist", "build", ".pytest_cache", ".mypy_cache"}


class RepoRefused(RuntimeError):
    pass


class RepoView:
    def __init__(self, root: str | Path, *, max_entries: int = 160, doc_files: int = 6,
                 doc_lines: int = 40) -> None:
        self.root = Path(root)
        self.max_entries = max_entries
        self.doc_files = doc_files
        self.doc_lines = doc_lines
        self.files = self._walk()

    def _refuse_transcripts(self) -> None:
        heavy = {".git", "node_modules", ".venv", "venv", "__pycache__"}
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel = Path(dirpath).relative_to(self.root)
            if rel.parts[-2:] == (".claude", "projects"):
                raise RepoRefused(f"{self.root} holds worker transcripts under .claude/projects")
            dirnames[:] = [d for d in dirnames if d not in heavy]
            for name in filenames:
                if name.endswith(".jsonl"):
                    raise RepoRefused(f"{self.root} holds a .jsonl file ({rel / name}); "
                                      "a worktree copy must not carry session transcripts")

    def _walk(self) -> list[str]:
        self._refuse_transcripts()
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel = Path(dirpath).relative_to(self.root)
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                files.append(str(rel / name) if str(rel) != "." else name)
        return files

    def tree(self) -> str:
        shown = self.files[: self.max_entries]
        more = len(self.files) - len(shown)
        return "\n".join(shown) + (f"\n[... {more} more files ...]" if more > 0 else "")

    def git_log(self, n: int = 15) -> str:
        if not (self.root / ".git").exists():
            return "(no commit history)"
        try:
            out = subprocess.run(
                ["git", "-C", str(self.root), "log", "--oneline", f"-n{n}"],
                capture_output=True, text=True, timeout=20,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        except (OSError, subprocess.TimeoutExpired):
            return "(no commit history)"
        return out.stdout.strip() or "(no commit history)"

    def docs(self) -> str:
        docs = [f for f in self.files if f.lower().endswith(".md")
                and f.count("/") <= 1][: self.doc_files]
        parts = []
        for rel in docs:
            lines = (self.root / rel).read_text(errors="replace").splitlines()[: self.doc_lines]
            parts.append(f"--- {rel} ---\n" + "\n".join(lines))
        return "\n\n".join(parts) or "(no documents)"

    def render(self) -> str:
        return (f"FILES:\n{self.tree()}\n\nCOMMITS (newest first):\n{self.git_log()}\n\n"
                f"DOCUMENTS (top of each):\n{self.docs()}")

    def unknown_paths(self, message: str) -> list[str]:
        """Path-like tokens in `message` that name no file or directory in the view."""
        known = set(self.files)
        dirs = {str(Path(f).parent) for f in self.files} | {
            str(p) for f in self.files for p in Path(f).parents if str(p) != "."}
        bad = []
        for tok in re.findall(r"`([^`\n]+)`|(\b[\w.-]+(?:/[\w.-]+)+\b)|(\b[\w-]+\.\w{1,5}\b)", message):
            token = next(t for t in tok if t)
            token = token.strip("./") if token.startswith("./") else token
            if "/" not in token and not re.search(r"\.\w{1,5}$", token):
                continue
            if token.startswith(("http", "www.")) or re.fullmatch(r"\d+(\.\d+)+", token):
                continue
            if token in known or token.rstrip("/") in dirs or any(
                    f.endswith("/" + token) for f in known):
                continue
            if re.search(r"\.(?:com|org|io|net|ai)$", token):
                continue
            if re.search(r"\.(?:py|js|ts|tsx|jsx|json|ya?ml|md|html|css|sql|sh|toml|txt|db)$", token):
                bad.append(token)
        return sorted(set(bad))
