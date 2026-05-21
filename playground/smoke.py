"""Smoke-test CLI — run a single chat turn end-to-end without Streamlit.

Usage:
    python -m playground.smoke --provider anthropic --model claude-sonnet-4-6 \
        --prompt "Hello"
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from playground.providers.base import ChatMessage, MessageComplete, TextBlock, TextDelta
from playground.providers.config import load_providers_config
from playground.providers.registry import get_client


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="playground.smoke")
    parser.add_argument("--provider", required=True, choices=["anthropic", "openai", "local"])
    parser.add_argument("--model", default=None, help="Defaults to providers.toml default_model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = load_providers_config()
    pcfg = cfg[args.provider]
    model = args.model or pcfg.default_model
    if not model:
        parser.error(f"--model required for {args.provider} (no default in providers.toml)")

    client = get_client(args.provider, model)
    events = client.stream_chat(
        messages=[ChatMessage(role="user", content=[TextBlock(type="text", text=args.prompt)])],
        system=args.system,
        tools=[],
        max_tokens=args.max_tokens or pcfg.default_max_tokens,
        temperature=args.temperature if args.temperature is not None else pcfg.default_temperature,
    )

    text_parts: list[str] = []
    final: MessageComplete | None = None
    for ev in events:
        if isinstance(ev, TextDelta):
            sys.stdout.write(ev.text)
            sys.stdout.flush()
            text_parts.append(ev.text)
        elif isinstance(ev, MessageComplete):
            final = ev
    sys.stdout.write("\n")

    if final:
        sys.stderr.write(
            f"[stop_reason={final.stop_reason} "
            f"in={final.usage.input_tokens} out={final.usage.output_tokens}]\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
