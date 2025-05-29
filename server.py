from mcp.server.fastmcp import FastMCP

from agentic_rag import AgenticRAG

# 初始化MCP服务器
mcp = FastMCP("agentic_rag_mcp_server")

@mcp.tool()
def query_document(
        user_query: str,
        document_path: str
) -> str:
    """
    查询文档
    """
    agent = AgenticRAG(document_path)

    agent.load_local_document()
    return user_query