"""mcp.py

Purpose:
- Model Context Protocol (MCP) is an open standard for connecting models to
  external tools, data, and prompts through a uniform server interface. Instead
  of hand-defining every tool, you point Claude at an MCP server and it gains
  that server's whole toolset.

Real-world applications:
- Connecting to GitHub, Linear, Asana, Google Drive, databases, or your own
  internal MCP server without writing bespoke tool schemas per integration.

When to use it:
- You already have (or can run) an MCP server exposing what you need and want
  standardized, reusable connections across projects.

Simple example:
- The MCP connector: declare a remote server + a matching mcp_toolset on the
  beta Messages API; Claude calls its tools server-side.

Production example:
- An allowlisted toolset over a remote server, plus the local-server pattern
  driven through the SDK Tool Runner.

Common mistakes:
- Declaring `mcp_servers` without a matching `mcp_toolset` in `tools` (400).
- `mcp_server_name` in the toolset not matching a server `name`.
- Forgetting the beta flag `mcp-client-2025-11-20`.
- Confusing the Messages-API MCP connector with Managed Agents (different API).

Best practices:
- Reference each declared server with exactly one toolset; use allowlist mode
  to expose only needed tools; keep credentials in `authorization_token`,
  never in the prompt.

Related concepts:
- tool_calling, function_calling, agents, guardrails.

Augmented Method learning cycle:
- Observe → Understand → Imitate → Modify → Predict → Build → Reflect → Teach
"""

from __future__ import annotations

import anthropic

MODEL = "claude-opus-4-8"
MCP_BETA = "mcp-client-2025-11-20"


def simple_example(user_input: str, server_url: str, token: str | None = None):
    """The connector needs BOTH halves: an mcp_servers entry AND a matching
    mcp_toolset in tools. Omitting the toolset is a 400."""
    client = anthropic.Anthropic()
    server = {"type": "url", "url": server_url, "name": "my-tools"}
    if token:
        server["authorization_token"] = token
    return client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=[MCP_BETA],
        mcp_servers=[server],
        tools=[{"type": "mcp_toolset", "mcp_server_name": "my-tools"}],
        messages=[{"role": "user", "content": user_input}],
    )


def production_example(user_input: str, server_url: str, token: str,
                       allowed_tools: list[str]):
    """Expose only specific tools from a large server (allowlist pattern)."""
    client = anthropic.Anthropic()
    return client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=[MCP_BETA],
        mcp_servers=[{"type": "url", "url": server_url, "name": "github",
                      "authorization_token": token}],
        tools=[{
            "type": "mcp_toolset",
            "mcp_server_name": "github",
            "default_config": {"enabled": False},           # deny by default
            "configs": [{"name": t, "enabled": True} for t in allowed_tools],
        }],
        messages=[{"role": "user", "content": user_input}],
    )


LOCAL_MCP_PATTERN = """\
For LOCAL MCP servers (stdio transport) the Python SDK converts MCP tools into
Tool Runner tools:

    from anthropic.lib.tools.mcp import mcp_tool       # async_mcp_tool for async
    from mcp import ClientSession
    # ... open a ClientSession to your local server, then:
    tools_result = mcp_client.list_tools()
    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-8", max_tokens=16000,
        tools=[mcp_tool(t, mcp_client) for t in tools_result.tools],
        messages=[{"role": "user", "content": "Use the available tools"}],
    )

Use the connector for remote HTTP servers; use these helpers for local servers.
"""


def main() -> None:
    print("mcp.py — Model Context Protocol client patterns")
    print("Connector requires beta flag:", MCP_BETA)
    print("Every mcp_servers entry needs a matching mcp_toolset in tools.\n")
    print(LOCAL_MCP_PATTERN)


if __name__ == "__main__":
    main()
