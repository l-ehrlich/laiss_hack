#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SYSTEM_PROMPT = """You are an MCP orchestration assistant.

You have access to MCP tools exposed by the connected server.
You must respond with exactly one JSON object and nothing else.

Allowed response shapes:
{"action":"call_tool","tool_name":"<tool name>","arguments":{"arg":"value"}}
{"action":"final","answer":"<final answer for the user>"}

Rules:
- Use tools whenever live or server-owned data is needed.
- Never invent tool names or tool arguments.
- After receiving a tool result, decide whether another tool call is needed or produce a final answer.
- Keep arguments strictly valid JSON.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MCP client for Chandra/XSA TAP workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("list-tools", "call-tool", "ask"):
        subparser = subparsers.add_parser(name)
        add_server_options(subparser)

    call_tool_parser = subparsers.choices["call-tool"]
    call_tool_parser.add_argument("--tool-name", required=True, help="Tool name to invoke.")
    call_tool_parser.add_argument(
        "--arguments",
        default="{}",
        help="Tool arguments as a JSON object string.",
    )

    ask_parser = subparsers.choices["ask"]
    ask_parser.add_argument(
        "--backend",
        required=True,
        choices=("openai", "anthropic", "gemini", "local"),
        help="LLM provider backend.",
    )
    ask_parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt to send through the LLM/MCP loop.",
    )
    ask_parser.add_argument(
        "--model",
        help="Override the model name. Otherwise the backend-specific environment variable is used.",
    )
    ask_parser.add_argument(
        "--base-url",
        help="Override the API base URL for OpenAI-compatible or hosted backends.",
    )
    ask_parser.add_argument(
        "--api-key",
        help="Override the API key instead of using the backend-specific environment variable.",
    )
    ask_parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        help="Maximum LLM/tool turns before the client aborts.",
    )

    return parser.parse_args()


def add_server_options(parser: argparse.ArgumentParser) -> None:
    default_server = Path(__file__).with_name("server.py")
    parser.add_argument(
        "--server-script",
        default=str(default_server),
        help="Path to the MCP server Python file.",
    )
    parser.add_argument(
        "--server-python",
        default=sys.executable,
        help="Python interpreter used to launch the MCP server.",
    )


def json_post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def ensure_value(value: str | None, label: str) -> str:
    if value:
        return value
    raise ValueError(f"Missing required configuration: {label}")


def render_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


class LLMBackend:
    def __init__(
        self,
        backend: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LLMBackend":
        if args.backend == "openai":
            return cls(
                backend="openai",
                model=ensure_value(args.model or os.getenv("OPENAI_MODEL"), "OPENAI_MODEL or --model"),
                base_url=args.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=ensure_value(args.api_key or os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY or --api-key"),
            )
        if args.backend == "anthropic":
            return cls(
                backend="anthropic",
                model=ensure_value(
                    args.model or os.getenv("ANTHROPIC_MODEL"),
                    "ANTHROPIC_MODEL or --model",
                ),
                base_url=args.base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
                api_key=ensure_value(
                    args.api_key or os.getenv("ANTHROPIC_API_KEY"),
                    "ANTHROPIC_API_KEY or --api-key",
                ),
            )
        if args.backend == "gemini":
            return cls(
                backend="gemini",
                model=ensure_value(args.model or os.getenv("GEMINI_MODEL"), "GEMINI_MODEL or --model"),
                base_url=args.base_url or os.getenv(
                    "GEMINI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta",
                ),
                api_key=ensure_value(args.api_key or os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY or --api-key"),
            )
        if args.backend == "local":
            return cls(
                backend="local",
                model=args.model or os.getenv("LOCAL_MODEL", "local-model"),
                base_url=args.base_url or os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key=args.api_key or os.getenv("LOCAL_API_KEY"),
            )
        raise ValueError(f"Unsupported backend: {args.backend}")

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.backend in {"openai", "local"}:
            return self._complete_openai_compatible(messages)
        if self.backend == "anthropic":
            return self._complete_anthropic(messages)
        if self.backend == "gemini":
            return self._complete_gemini(messages)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def _complete_openai_compatible(self, messages: list[dict[str, str]]) -> str:
        assert self.base_url is not None
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        data = json_post(f"{self.base_url.rstrip('/')}/chat/completions", headers, payload)
        return render_openai_content(data["choices"][0]["message"]["content"])

    def _complete_anthropic(self, messages: list[dict[str, str]]) -> str:
        assert self.base_url is not None
        system_blocks = [message["content"] for message in messages if message["role"] == "system"]
        anthropic_messages = [
            {
                "role": "assistant" if message["role"] == "assistant" else "user",
                "content": [{"type": "text", "text": message["content"]}],
            }
            for message in messages
            if message["role"] != "system"
        ]
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": "\n\n".join(system_blocks),
            "messages": anthropic_messages,
        }
        headers = {
            "x-api-key": ensure_value(self.api_key, "Anthropic API key"),
            "anthropic-version": "2023-06-01",
        }
        data = json_post(f"{self.base_url.rstrip('/')}/messages", headers, payload)
        parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)

    def _complete_gemini(self, messages: list[dict[str, str]]) -> str:
        assert self.base_url is not None
        api_key = ensure_value(self.api_key, "Gemini API key")
        system_text = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        contents = []
        for message in messages:
            if message["role"] == "system":
                continue
            contents.append(
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": [{"text": message["content"]}],
                }
            )
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": 0},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        model_name = urllib.parse.quote(self.model, safe="")
        url = f"{self.base_url.rstrip('/')}/models/{model_name}:generateContent?key={api_key}"
        data = json_post(url, {}, payload)
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data, indent=2)}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and "text" in part
        )


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {text}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError(f"Unterminated JSON object in model response: {text}")


def serialize_tool(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(mode="json")
    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "inputSchema": getattr(tool, "inputSchema", None),
    }


def normalize_call_tool_result(result: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "is_error": getattr(result, "isError", False),
        "structured_content": getattr(result, "structuredContent", None),
        "content": [],
    }
    for item in getattr(result, "content", []):
        if hasattr(item, "model_dump"):
            normalized["content"].append(item.model_dump(mode="json"))
        else:
            normalized["content"].append(str(item))
    return normalized


def build_initial_user_message(prompt: str, tools: list[dict[str, Any]]) -> str:
    return (
        "User request:\n"
        f"{prompt}\n\n"
        "Available MCP tools:\n"
        f"{json.dumps(tools, indent=2)}\n\n"
        "Reply with one JSON object following the required schema."
    )


def build_tool_result_message(tool_name: str, result_payload: dict[str, Any]) -> str:
    return (
        f"Tool `{tool_name}` returned:\n"
        f"{json.dumps(result_payload, indent=2)}\n\n"
        "Reply with either another tool call JSON object or a final answer JSON object."
    )


def get_server_parameters(args: argparse.Namespace) -> StdioServerParameters:
    server_script = Path(args.server_script).resolve()
    return StdioServerParameters(
        command=args.server_python,
        args=[str(server_script)],
        cwd=str(server_script.parent),
    )


async def open_session(args: argparse.Namespace) -> tuple[ClientSession, Any]:
    server_params = get_server_parameters(args)
    client_cm = stdio_client(server_params)
    streams = await client_cm.__aenter__()
    read_stream, write_stream = streams
    session_cm = ClientSession(read_stream, write_stream)
    session = await session_cm.__aenter__()
    try:
        await session.initialize()
    except Exception:
        await session_cm.__aexit__(*sys.exc_info())
        await client_cm.__aexit__(*sys.exc_info())
        raise
    return session, (session_cm, client_cm)


async def close_session(resources: tuple[Any, Any]) -> None:
    session_cm, client_cm = resources
    await session_cm.__aexit__(None, None, None)
    await client_cm.__aexit__(None, None, None)


async def handle_list_tools(args: argparse.Namespace) -> int:
    session, resources = await open_session(args)
    try:
        tools = await session.list_tools()
        print(json.dumps([serialize_tool(tool) for tool in tools.tools], indent=2))
        return 0
    finally:
        await close_session(resources)


async def handle_call_tool(args: argparse.Namespace) -> int:
    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--arguments must be valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise SystemExit("--arguments must decode to a JSON object.")

    session, resources = await open_session(args)
    try:
        result = await session.call_tool(args.tool_name, arguments)
        print(json.dumps(normalize_call_tool_result(result), indent=2))
        return 0
    finally:
        await close_session(resources)


async def handle_ask(args: argparse.Namespace) -> int:
    backend = LLMBackend.from_args(args)
    session, resources = await open_session(args)
    try:
        tool_result = await session.list_tools()
        tools = [serialize_tool(tool) for tool in tool_result.tools]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_message(args.prompt, tools)},
        ]

        for _ in range(args.max_iterations):
            raw_response = backend.complete(messages)
            messages.append({"role": "assistant", "content": raw_response})
            try:
                action = extract_json_object(raw_response)
            except ValueError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed: {exc}. "
                            "Reply again with a single valid JSON object only."
                        ),
                    }
                )
                continue

            action_type = action.get("action")
            if action_type == "final":
                answer = action.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The `final` action requires a non-empty string in the `answer` field. "
                                "Reply again with a valid JSON object only."
                            ),
                        }
                    )
                    continue
                print(answer)
                return 0

            if action_type == "call_tool":
                tool_name = action.get("tool_name")
                arguments = action.get("arguments", {})
                if not isinstance(tool_name, str) or not tool_name:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The `call_tool` action requires a non-empty string in `tool_name`. "
                                "Reply again with a valid JSON object only."
                            ),
                        }
                    )
                    continue
                if not isinstance(arguments, dict):
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The `call_tool` action requires `arguments` to be a JSON object. "
                                "Reply again with a valid JSON object only."
                            ),
                        }
                    )
                    continue
                tool_call_result = await session.call_tool(tool_name, arguments)
                messages.append(
                    {
                        "role": "user",
                        "content": build_tool_result_message(
                            tool_name,
                            normalize_call_tool_result(tool_call_result),
                        ),
                    }
                )
                continue

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The `action` field must be either `call_tool` or `final`. "
                        "Reply again with a valid JSON object only."
                    ),
                }
            )

        raise RuntimeError(
            f"Reached the max iteration limit ({args.max_iterations}) without a final answer."
        )
    finally:
        await close_session(resources)


async def async_main() -> int:
    args = parse_args()
    if args.command == "list-tools":
        return await handle_list_tools(args)
    if args.command == "call-tool":
        return await handle_call_tool(args)
    if args.command == "ask":
        return await handle_ask(args)
    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
