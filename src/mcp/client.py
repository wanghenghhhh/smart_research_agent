import asyncio
import logging
from typing import Any, Dict, List, Optional
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

logger = logging.getLogger(__name__)

class MCPClient:
    """MCP 客户端封装，管理与 MCP Server 的 SSE 长连接"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8000/sse"):
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self._sse_ctx = None
        self._session_ctx = None

    async def connect(self):
        """建立与 MCP Server 的长连接"""
        if self.session:
            return
        
        print("🔌 正在与 MCP Server 建立长连接...")
        try:
            # 初始化 SSE 客户端与 Session 上下文
            self._sse_ctx = sse_client(self.server_url)
            read_stream, write_stream = await self._sse_ctx.__aenter__()
            
            self._session_ctx = ClientSession(read_stream, write_stream)
            self.session = await self._session_ctx.__aenter__()
            
            # 初始化协议
            await self.session.initialize()
            print("✅ MCP Server 连接成功！")
        except Exception as e:
            await self.close()
            raise RuntimeError(f"连接 MCP Server 失败: {e}")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """获取 MCP 服务端提供的所有工具列表"""
        if not self.session:
            await self.connect()
        
        response = await self.session.list_tools()
        tools_data = []
        for tool in response.tools:
            tools_data.append({
                "name": tool.name,
                "description": tool.description or f"MCP 工具: {tool.name}",
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {}
            })
        return tools_data

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected or not self.session:
            raise RuntimeError("MCP Client 未连接")

        try:
            response = await asyncio.wait_for(
                self.session.call_tool(name, arguments),
                timeout=10.0
            )
            if response.content:
                return response.content[0].text
            return "工具执行成功，但无文本返回。"
        except asyncio.TimeoutError:
            return f"❌ MCP 工具 {name} 执行超时（10秒）"
        except Exception as e:
            return f"❌ 工具调用失败: {str(e)}"

    async def close(self):
        """安全释放 MCP 连接资源"""
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
            self.session = None

        if self._sse_ctx:
            try:
                await self._sse_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._sse_ctx = None


# ============================================================
# 全局单例管理与 Adapter 适配器函数
# ============================================================

_global_mcp_client: Optional[MCPClient] = None

async def get_mcp_client(server_url: str = "http://127.0.0.1:8000/sse") -> MCPClient:
    """获取全局唯一的 MCPClient 实例（保持长连接）"""
    global _global_mcp_client
    if _global_mcp_client is None:
        _global_mcp_client = MCPClient(server_url)
        await _global_mcp_client.connect()
    return _global_mcp_client


def create_mcp_tool_adapter(client: MCPClient, tool_info: Dict[str, Any]) -> StructuredTool:
    """
    将 MCP 的 Tool 信息转换为 LangChain 兼容的 StructuredTool
    必须包含 docstring，保证兼容性
    """
    tool_name = tool_info["name"]
    # 保证 description 绝不为空，并作为兜底文档说明
    tool_desc = tool_info.get("description") or f"MCP 工具: {tool_name}"
    input_schema = tool_info.get("input_schema") or {}

    # 1. 提取 json schema 并构建 Pydantic 模型
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))
    
    fields = {}
    for prop_name, prop_info in properties.items():
        field_type = str
        field_desc = prop_info.get("description", "")
        default_value = ... if prop_name in required_fields else None
        fields[prop_name] = (field_type, Field(default=default_value, description=field_desc))

    # 动态参数模型
    ArgsModel = create_model(f"{tool_name}_input", **fields) if fields else None

    # 2. 包装异步调用闭包函数
    async def _run_tool(**kwargs) -> str:
        return await client.call_tool(tool_name, kwargs)

    # 关键修复：显式写入 __doc__，规避 LangChain 校验失败
    _run_tool.__doc__ = tool_desc

    # 3. 构造 StructuredTool 实例
    return StructuredTool.from_function(
        coroutine=_run_tool,
        name=tool_name,
        description=tool_desc,
        args_schema=ArgsModel
    )

async def close_mcp_client():
    """程序退出时显式关闭 MCP 连接"""
    global _global_mcp_client
    if _global_mcp_client is not None:
        await _global_mcp_client.close()
        _global_mcp_client = None
        print("🔌 MCP 连接已安全关闭")