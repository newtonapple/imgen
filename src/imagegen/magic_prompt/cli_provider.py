# src/imagegen/magic_prompt/cli_provider.py
"""Magic-prompt via the codex/pi CLI (ports the ComfyUI CodexPromptToJson node)."""
from __future__ import annotations
import json
import re
import shutil
import subprocess
from pathlib import Path

from ..engine.resolution import aspect_ratio

_TEMPLATE = (Path(__file__).parent / "templates" / "ideogram4_caption.txt").read_text()

CODEX_MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
PI_MODELS_PATH = "~/.pi/agent/models.json"


class CliMagicPromptProvider:
    def __init__(self, model: str = "codex - gpt-5.5", timeout_s: int = 120, runner=subprocess.run):
        self.model = model
        self.timeout_s = timeout_s
        self._runner = runner

    def expand(self, prompt: str, *, width: int, height: int, target_elements: int = 0) -> dict:
        ar = aspect_ratio(width, height)
        instruction = f"{_TEMPLATE}\n\nUSER PROMPT:\n{prompt}\n\n[ASPECT RATIO]\nReturn aspect_ratio exactly \"{ar}\" and compose bbox for a {width}x{height} canvas."
        if target_elements > 0:
            instruction += (f"\n\n[ELEMENT COUNT]\nReturn a target of exactly {target_elements} "
                            "entries in compositional_deconstruction.elements (add real subjects/"
                            "props/text, not body parts); include every explicit text element even if it exceeds the target.")
        cmd = self._build_cmd(instruction)
        result = self._runner(cmd, capture_output=True, text=True, timeout=self.timeout_s, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"magic-prompt CLI failed: {(result.stderr or result.stdout)[:300]}")
        parsed = json.loads(self._extract_json(result.stdout))
        parsed["aspect_ratio"] = ar
        return parsed

    def _build_cmd(self, instruction: str) -> list[str]:
        if self.model.startswith("pi - "):
            provider, _, model_id = self.model.removeprefix("pi - ").partition(" - ")
            pi = shutil.which("pi") or "pi"
            return [pi, "--provider", provider, "--model", model_id, "--print",
                    "--no-tools", "--no-context-files", "--no-session", "--thinking", "off", instruction]
        model_id = self.model.removeprefix("codex - ").strip()
        codex = shutil.which("codex") or "codex"
        return [codex, "exec", "--model", model_id, "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", instruction]

    @staticmethod
    def available_models() -> list[str]:
        """Codex models + text-capable pi models from ~/.pi/agent/models.json
        (override path via PI_MODELS_JSON env var). Mirrors the ComfyUI CodexPromptToJson node."""
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

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        return text[start:end + 1] if start != -1 and end != -1 and end > start else text
