#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "chandra-xsa-tap"
DEFAULT_TAP_ENDPOINT = "https://example.invalid/chandra/xsa/tap"

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "Expose Chandra/XSA TAP retrieval tools over MCP. The current TAP query "
        "implementation is a placeholder so the tool contract can be wired in now "
        "and replaced with the real transport later."
    ),
    log_level="WARNING",
)


def run_chandra_tap_query(adql: str, max_rows: int = 100) -> dict[str, Any]:
    endpoint = os.getenv("CHANDRA_TAP_ENDPOINT", DEFAULT_TAP_ENDPOINT)
    return {
        "status": "placeholder",
        "message": (
            "Replace `run_chandra_tap_query` in server.py with your real XSA TAP "
            "query implementation."
        ),
        "tap_endpoint": endpoint,
        "adql": adql,
        "max_rows": max_rows,
        "notes": [
            "This MCP tool contract is ready to be consumed by an MCP client.",
            "The returned payload shape is intended to stay stable when you swap in the real TAP call.",
            "Set CHANDRA_TAP_ENDPOINT to point at your target TAP service.",
        ],
        "rows": [],
    }


@mcp.tool(
    name="query_chandra_tap",
    description=(
        "Run an ADQL query against a Chandra/XSA TAP service and return rows plus "
        "basic request metadata. This is a placeholder implementation for now."
    ),
    structured_output=True,
)
def query_chandra_tap(adql: str, max_rows: int = 100) -> dict[str, Any]:
    """Execute a placeholder TAP query until the real Chandra/XSA backend is wired in."""
    return run_chandra_tap_query(adql=adql, max_rows=max_rows)


if __name__ == "__main__":
    mcp.run(transport="stdio")
