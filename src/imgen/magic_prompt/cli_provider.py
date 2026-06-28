# src/imgen/magic_prompt/cli_provider.py
"""Magic-prompt via a local CLI (codex / claude / pi) — no API key needed."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from ._caption_prompt import run_magic_prompt

CODEX_MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
PI_MODELS_PATH = "~/.pi/agent/models.json"


class CliMagicPromptProvider:
    """Runs a one-shot CLI tool and parses its stdout into a caption dict."""

    def __init__(
        self,
        provider: str = "codex",
        model: str = "gpt-5.5",
        timeout_s: int = 120,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ):
        self.provider = provider
        self.model = model
        self.timeout_s = timeout_s
        self._runner = runner

    def expand(
        self, prompt: str, *, width: int, height: int, target_elements: int = 0
    ) -> dict[str, Any]:
        return run_magic_prompt(
            self._call_model,
            prompt=prompt,
            width=width,
            height=height,
            target_elements=target_elements,
        )

    def _call_model(self, instruction: str) -> str:
        result = self._runner(
            self._build_cmd(instruction),
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"magic-prompt CLI failed: {(result.stderr or result.stdout)[:300]}")
        return str(result.stdout)

    def _build_cmd(self, instruction: str) -> list[str]:
        if self.provider == "pi":
            pi_provider, _, model_id = self.model.partition("/")
            pi = shutil.which("pi") or "pi"
            return [
                pi,
                "--provider",
                pi_provider,
                "--model",
                model_id,
                "--print",
                "--no-tools",
                "--no-context-files",
                "--no-session",
                "--thinking",
                "off",
                instruction,
            ]
        if self.provider == "claude":
            claude = shutil.which("claude") or "claude"
            # Headless, non-interactive (-p) completion with NO tools. -p already
            # denies tools that need approval (nothing can grant it), and we
            # explicitly deny the filesystem/shell/web/subagent tools as defense
            # in depth so an injected prompt can't read files or run commands.
            return [
                claude,
                "-p",
                instruction,
                "--model",
                self.model,
                "--disallowedTools",
                "Bash,Edit,Write,MultiEdit,NotebookEdit,Read,Glob,Grep,WebFetch,WebSearch,Task",
            ]
        codex = shutil.which("codex") or "codex"
        # read-only sandbox (no writes, no network) + --ignore-user-config so a
        # trusted-project entry in ~/.codex/config.toml can't relax the sandbox;
        # --ephemeral keeps no session state. codex still exposes a shell tool,
        # but read-only + network isolation mean an injected prompt cannot write
        # or exfiltrate (auth still resolves from CODEX_HOME).
        return [
            codex,
            "exec",
            "--model",
            self.model,
            "--sandbox",
            "read-only",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--ephemeral",
            instruction,
        ]

    @staticmethod
    def available_models() -> list[str]:
        """Reference list: codex models + text-capable pi models from
        ~/.pi/agent/models.json (override path via PI_MODELS_JSON)."""
        import os

        choices = [f"codex - {m}" for m in CODEX_MODELS]
        path = os.path.expanduser(os.environ.get("PI_MODELS_JSON", PI_MODELS_PATH))
        try:
            with open(path) as f:
                config = json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            return choices
        for provider_name, provider in config.get("providers", {}).items():
            for model in provider.get("models", []):
                model_id = model.get("id") if isinstance(model, dict) else None
                if not model_id:
                    continue
                input_types = model.get("input")
                if input_types and "text" not in input_types:
                    continue
                choices.append(f"pi - {provider_name} - {model_id}")
        return choices
