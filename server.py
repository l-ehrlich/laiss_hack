#!/usr/bin/env python3
from __future__ import annotations
import json
import tempfile
from functools import lru_cache
from typing import TypedDict
import os
from pathlib import Path
from typing import Any
import pyvo as vo
from mcp.server.fastmcp import FastMCP
from astroquery.esa.xmm_newton import XMMNewton
from uuid import uuid4

TABLE_NAME = "csc21.observation_source"
SERVER_NAME = "chandra-xsa-tap"
DEFAULT_TAP_ENDPOINT = "https://example.invalid/chandra/xsa/tap"

ADQL_EXAMPLES = [
    {
        "user_query": "Find 10 sources from observation_source.",
        "tap_query": "SELECT TOP 10 * FROM csc21.observation_source"
    },
    {
        "user_query": "Show the ra and dec of 5 rows in observation_source.",
        "tap_query": "SELECT TOP 5 ra, dec FROM csc21.observation_source"
    },
]

chandra_mcp = FastMCP(
    name="chandra-csc-tap",
    instructions=(
        "Chandra Source Catalog (CSC 2.1) archive tools. "
        "Query the Chandra TAP service for X-ray source detections, "
        "observation metadata, and catalog columns."
    ),
    host="0.0.0.0",
    port=8001,
    log_level="WARNING",
)

xmm_mcp = FastMCP(
    name="xmm-newton-xsa-tap",
    instructions=(
        "XMM-Newton Science Archive (XSA) tools. "
        "Query the XMM-Newton TAP service for X-ray observations, "
        "source detections, and catalog metadata."
    ),
    host="0.0.0.0",
    port=8002,
    log_level="WARNING",
)

class ColumnsResult(TypedDict):
    table_name: str
    column_count: int
    column_names: list[str]


class ColumnMetadataRow(TypedDict):
    column_name: str
    datatype: str | None
    unit: str | None
    ucd: str | None
    utype: str | None
    description: str | None
    indexed: int | bool | None
    principal: int | bool | None
    std: int | bool | None


class ColumnMetadataResult(TypedDict):
    table_name: str
    column_count: int
    columns: list[ColumnMetadataRow]


tap = vo.dal.TAPService("https://cda.cfa.harvard.edu/csc21tap")

@chandra_mcp.tool()
def list_all_tables() -> dict:
    """
    Return all available table names from the TAP service.
    No input arguments required.
    """
    table_names = sorted(tap.tables.keys())

    return {
        "table_count": len(table_names),
        "table_names": table_names,
    }

@chandra_mcp.tool()
def get_table_columns(table_name) -> dict:
    """
    Return all column names for input table.
    """
    table = tap.tables[table_name]
    column_names = [col.name for col in table.columns]

    return {
        "table_name": table_name,
        "column_count": len(column_names),
        "column_names": column_names,
    }

@chandra_mcp.tool()
def get_table_column_metadata(table_name) -> dict:
    """
    Return detailed column metadata for input table.
    """
    query = f"""
    SELECT
        table_name,
        column_name,
        datatype,
        unit,
        ucd,
        utype,
        description,
        indexed,
        principal,
        std
    FROM TAP_SCHEMA.columns
    WHERE table_name = '{table_name}'
    ORDER BY column_name
    """

    results = tap.search(query)

    columns = []
    for row in results:
        columns.append({
            "column_name": row["column_name"],
            "datatype": row["datatype"],
            "unit": row["unit"],
            "ucd": row["ucd"],
            "utype": row["utype"],
            "description": row["description"],
            "indexed": row["indexed"],
            "principal": row["principal"],
            "std": row["std"],
        })

    return {
        "table_name": table_name,
        "column_count": len(columns),
        "columns": columns,
    }

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

@chandra_mcp.tool()
def get_adql_examples() -> dict:
    """
    Return example user queries paired with correct TAP/ADQL queries.
    This is intended to ground the agent in how natural-language requests
    map to valid ADQL for the TAP service.
    """
    return {
        "example_count": len(ADQL_EXAMPLES),
        "examples": ADQL_EXAMPLES,
    }

def _jsonify(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value

def _astropy_table_to_rows(table) -> tuple[list[str], list[dict[str, Any]]]:
    columns = list(table.colnames)
    rows = []
    for record in table:
        rows.append({col: _jsonify(record[col]) for col in columns})
    return columns, rows

@xmm_mcp.tool()
def list_all_xmm_tables() -> dict:
    """
    Return all available table names from the XMM-Newton XSA TAP service.
    """
    table_names = sorted(str(t) for t in XMMNewton.get_tables())
    return {
        "mission": "xmm_newton",
        "table_count": len(table_names),
        "table_names": table_names,
    }

@xmm_mcp.tool()
def get_xmm_table_columns(table_name: str) -> dict:
    """
    Return all column names for an XMM-Newton XSA TAP table.
    """
    column_names = list(XMMNewton.get_columns(table_name, only_names=True))
    return {
        "mission": "xmm_newton",
        "table_name": table_name,
        "column_count": len(column_names),
        "column_names": column_names,
    }

@xmm_mcp.tool()
def get_xmm_table_column_metadata(table_name: str) -> dict:
    """
    Return detailed column metadata for an XMM-Newton XSA TAP table.
    """
    columns = XMMNewton.get_columns(table_name, only_names=False)

    serialized = []
    for col in columns:
        serialized.append({
            "column_name": getattr(col, "name", None),
            "datatype": getattr(col, "datatype", None),
            "unit": getattr(col, "unit", None),
            "ucd": getattr(col, "ucd", None),
            "utype": getattr(col, "utype", None),
            "description": getattr(col, "description", None),
            "indexed": getattr(col, "indexed", None),
            "principal": getattr(col, "principal", None),
            "std": getattr(col, "std", None),
        })

    return {
        "mission": "xmm_newton",
        "table_name": table_name,
        "column_count": len(serialized),
        "columns": serialized,
    }

@xmm_mcp.tool(
    name="query_xmm_tap",
    description="Run an ADQL query against the XMM-Newton XSA TAP service and return a preview.",
    structured_output=True,
)
def query_xmm_tap(adql: str) -> dict[str, Any]:
    table = XMMNewton.query_xsa_tap(adql, output_format="votable")
    columns, rows = _astropy_table_to_rows(table)

    return {
        "mission": "xmm_newton",
        "adql": adql,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "preview_only": True,
    }

@xmm_mcp.tool(
    name="export_xmm_tap_jsonl",
    description="Run an ADQL query against the XMM-Newton XSA TAP service and save the full result as JSONL.",
    structured_output=True,
)
def export_xmm_tap_jsonl(adql: str) -> dict[str, Any]:
    table = XMMNewton.query_xsa_tap(adql, output_format="votable")
    columns = list(table.colnames)

    output_dir = Path(tempfile.gettempdir()) / "xmm_mcp_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"xmm_tap_result_{uuid4().hex}.jsonl"

    row_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in table:
            row = {col: _jsonify(record[col]) for col in columns}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1

    return {
        "mission": "xmm_newton",
        "status": "ok",
        "adql": adql,
        "row_count": row_count,
        "columns": columns,
        "file_path": str(output_path),
        "file_name": output_path.name,
        "format": "jsonl",
    }



@chandra_mcp.tool(
    name="query_chandra_tap",
    description=(
        "Run an ADQL query against the Chandra TAP service. "
        "Returns ALL matching rows by default (max_rows=10000). "
        "The UI renders results as a sortable, downloadable table. "
        "Do NOT add TOP or LIMIT to the ADQL unless the user explicitly asks for a subset. "
        "For result sets larger than 10000 rows, use export_chandra_tap_jsonl instead."
    ),
    structured_output=True,
)
def query_chandra_tap(adql: str, max_rows: int = 10000) -> dict[str, Any]:
    results = tap.search(adql, maxrec=max_rows)
    columns = list(results.fieldnames)

    rows = []
    for record in results:
        rows.append({col: _jsonify(record[col]) for col in columns})

    truncated = len(rows) >= max_rows

    return {
        "adql": adql,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "truncated": truncated,
    }


@chandra_mcp.tool(
    name="export_chandra_tap_jsonl",
    description="Run an ADQL query and save the full result as JSONL.",
    structured_output=True,
)
def export_chandra_tap_jsonl(adql: str, max_rows: int = 50000) -> dict[str, Any]:
    results = tap.run_async(adql, maxrec=max_rows)
    columns = list(results.fieldnames)

    output_dir = Path(tempfile.gettempdir()) / "chandra_mcp_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "tap_result.jsonl"

    row_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            row = {col: _jsonify(record[col]) for col in columns}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            row_count += 1

    return {
        "status": "ok",
        "adql": adql,
        "row_count": row_count,
        "columns": columns,
        "file_path": str(output_path),
        "file_name": output_path.name,
        "format": "jsonl",
    }


if __name__ == "__main__":
    import sys
    import asyncio
    import uvicorn

    args = set(sys.argv[1:])

    if "--streaming" in args:
        # Run both servers concurrently
        async def run_both():
            chandra_app = chandra_mcp.streamable_http_app()
            xmm_app = xmm_mcp.streamable_http_app()

            config_c = uvicorn.Config(chandra_app, host="0.0.0.0", port=8001, log_level="warning")
            config_x = uvicorn.Config(xmm_app, host="0.0.0.0", port=8002, log_level="warning")

            server_c = uvicorn.Server(config_c)
            server_x = uvicorn.Server(config_x)

            print("Chandra MCP on port 8001, XMM-Newton MCP on port 8002")
            await asyncio.gather(server_c.serve(), server_x.serve())

        asyncio.run(run_both())
    elif "--chandra" in args:
        chandra_mcp.run(transport="streamable-http")
    elif "--xmm" in args:
        xmm_mcp.run(transport="streamable-http")
    else:
        chandra_mcp.run(transport="stdio")
