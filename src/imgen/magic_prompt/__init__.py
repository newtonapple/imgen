from .base import MagicPromptProvider
from .cli_provider import CliMagicPromptProvider
from .http_provider import HttpMagicPromptProvider
from .ideogram_provider import IdeogramMagicPromptProvider
from .providers import make_magic_provider, resolve_magic_provider

__all__ = [
    "MagicPromptProvider",
    "CliMagicPromptProvider",
    "HttpMagicPromptProvider",
    "IdeogramMagicPromptProvider",
    "make_magic_provider",
    "resolve_magic_provider",
]
