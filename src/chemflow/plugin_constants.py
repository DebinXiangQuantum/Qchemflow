"""Agent Plugins stdio command/args fields require verification on installation.

The documented schema is used for generated mcp.json; installed framework
releases may differ in their acceptance of these stdio field names.
"""

AGENT_PLUGIN_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
AGENT_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"
STDIO_FIELDS = ("type", "command", "args")
