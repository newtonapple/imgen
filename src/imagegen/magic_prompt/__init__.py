from .base import MagicPromptProvider
from .cli_provider import CliMagicPromptProvider
from .http_provider import HttpMagicPromptProvider
from .providers import make_magic_provider, resolve_magic_provider

__all__ = [
    "MagicPromptProvider",
    "CliMagicPromptProvider",
    "HttpMagicPromptProvider",
    "make_magic_provider",
    "resolve_magic_provider",
]
