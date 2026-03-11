#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


OPENAI_MODEL = "gpt-5-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1"
SERVER_SCRIPT = Path(__file__).with_name("server.py")
SERVER_PYTHON = sys.executable

SYSTEM_PROMPT = """You are an MCP orchestration assistant.

You have access to MCP tools exposed by the connected server.
You must respond with exactly one JSON object and nothing else.

Allowed response shapes:
{"action":"call_tool","tool_name":"<tool name>","arguments":{"arg":"value"}}
{"action":"final","answer":"<final answer for the user>"}

Rules:
- Use tools whenever live or server-owned data is needed.
- Never invent tool names or tool arguments.
- For schema exploration tasks, prefer calling tools before answering.
- If the user asks about tables, inspect tables first.
- If the user asks about columns in a table, inspect the table columns first.
- After receiving a tool result, decide whether another tool call is needed or produce a final answer.
- Keep arguments strictly valid JSON.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MCP client using OpenAI GPT-5 mini.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tools")

    call_tool_parser = subparsers.add_parser("call-tool")
    call_tool_parser.add_argument("--tool-name", required=True)
    call_tool_parser.add_argument("--arguments", default="{}")

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("--prompt", required=True)
    ask_parser.add_argument("--max-iterations", type=int, default=8)

    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = ""
    if api_key:
        return api_key
    raise ValueError("Missing OPENAI_API_KEY environment variable.")


def log(message: str) -> None:
    print(message, flush=True)


def preview(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def json_post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def render_openai_responses_content(data: dict[str, Any]) -> str:
    parts: list[str] = []

    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue

        for content_item in item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                parts.append(content_item.get("text", ""))
            elif content_item.get("type") == "refusal":
                parts.append(content_item.get("refusal", ""))

    text = "".join(parts).strip()
    if text:
        return text

    raise RuntimeError(f"OpenAI Responses API returned no assistant text: {json.dumps(data, indent=2)}")


def complete_with_openai(messages: list[dict[str, str]]) -> str:
    headers = {"Authorization": f"Bearer {ensure_api_key()}"}
    payload = {
        "model": OPENAI_MODEL,
        "input": [{"role": m["role"], "content": m["content"]} for m in messages],
        "reasoning": {"effort": "low"},
    }
    data = json_post(f"{OPENAI_BASE_URL}/responses", headers, payload)
    return render_openai_responses_content(data)


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
                return json.loads(text[start:index + 1])

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


def get_server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=SERVER_PYTHON,
        args=[str(SERVER_SCRIPT.resolve())],
        cwd=str(SERVER_SCRIPT.resolve().parent),
    )


async def open_session() -> tuple[ClientSession, Any]:
    client_cm = stdio_client(get_server_parameters())
    read_stream, write_stream = await client_cm.__aenter__()

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


async def handle_list_tools() -> int:
    session, resources = await open_session()
    try:
        tools = await session.list_tools()
        print(json.dumps([serialize_tool(tool) for tool in tools.tools], indent=2))
        return 0
    finally:
        await close_session(resources)


async def handle_call_tool(args: argparse.Namespace) -> int:
    arguments = json.loads(args.arguments)
    if not isinstance(arguments, dict):
        raise SystemExit("--arguments must decode to a JSON object.")

    session, resources = await open_session()
    try:
        result = await session.call_tool(args.tool_name, arguments)
        print(json.dumps(normalize_call_tool_result(result), indent=2))
        return 0
    finally:
        await close_session(resources)


async def handle_ask(args: argparse.Namespace) -> int:
    session, resources = await open_session()
    try:
        tool_result = await session.list_tools()
        tools = [serialize_tool(tool) for tool in tool_result.tools]

        log(f"\nUSER PROMPT:\n{args.prompt}\n")
        log(f"TOOLS AVAILABLE: {[tool['name'] for tool in tools]}\n")

        messages: list[dict[str, str]] = [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_initial_user_message(args.prompt, tools)},
        ]

        for step in range(1, args.max_iterations + 1):
            log(f"--- ITERATION {step} ---")
            raw_response = complete_with_openai(messages)
            log(f"MODEL RAW RESPONSE:\n{raw_response}\n")
            messages.append({"role": "assistant", "content": raw_response})

            try:
                action = extract_json_object(raw_response)
            except ValueError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response could not be parsed: {exc}. Reply again with a single valid JSON object only.",
                    }
                )
                continue

            if action.get("action") == "final":
                answer = action.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    messages.append(
                        {
                            "role": "user",
                            "content": "The `final` action requires a non-empty string in the `answer` field. Reply again with a valid JSON object only.",
                        }
                    )
                    continue

                log(f"FINAL ANSWER:\n{answer}\n")
                print(answer)
                return 0

            if action.get("action") == "call_tool":
                tool_name = action.get("tool_name")
                arguments = action.get("arguments", {})

                if not isinstance(tool_name, str) or not tool_name:
                    messages.append(
                        {
                            "role": "user",
                            "content": "The `call_tool` action requires a non-empty string in `tool_name`. Reply again with a valid JSON object only.",
                        }
                    )
                    continue

                if not isinstance(arguments, dict):
                    messages.append(
                        {
                            "role": "user",
                            "content": "The `call_tool` action requires `arguments` to be a JSON object. Reply again with a valid JSON object only.",
                        }
                    )
                    continue

                log(f"CALLING TOOL: {tool_name}")
                log(f"ARGUMENTS:\n{preview(arguments)}\n")

                tool_call_result = await session.call_tool(tool_name, arguments)
                normalized_result = normalize_call_tool_result(tool_call_result)

                log(f"TOOL RESULT:\n{preview(normalized_result)}\n")

                messages.append(
                    {
                        "role": "user",
                        "content": build_tool_result_message(tool_name, normalized_result),
                    }
                )
                continue

            messages.append(
                {
                    "role": "user",
                    "content": "The `action` field must be either `call_tool` or `final`. Reply again with a valid JSON object only.",
                }
            )

        raise RuntimeError(f"Reached the max iteration limit ({args.max_iterations}) without a final answer.")
    finally:
        await close_session(resources)


async def async_main() -> int:
    args = parse_args()
    if args.command == "list-tools":
        return await handle_list_tools()
    if args.command == "call-tool":
        return await handle_call_tool(args)
    if args.command == "ask":
        return await handle_ask(args)
    raise SystemExit(f"Unsupported command: {args.command}")


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())