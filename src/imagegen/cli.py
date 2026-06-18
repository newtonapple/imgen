"""CLI entrypoint (skeleton).

`imagegen platform` reports the detected platform + default backend — useful to
confirm the right deps are installed. magic-prompt / generate / run / serve land
here as the pipeline fills in.
"""

from __future__ import annotations

import argparse
import json
import sys

from .platform import platform_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="imagegen")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("platform", help="show detected platform + default backend")

    args = parser.parse_args(argv)

    if args.cmd == "platform":
        json.dump(platform_summary(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
