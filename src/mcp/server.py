import asyncio
import os
import subprocess
from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
import mcp.types as types

server = FastMCP("smart-research-mcp-server")

@server.tool()
async def read_file(path: str) -> str:
    """
    读取本地文件内容。
    输入参数：path - 文件的绝对路径或相对路径。
    返回文件内容的文本。
    """
    try:
        # 安全检查：防止读取敏感系统文件（演示用，生产需加强）
        if "/etc/passwd" in path or "C:\\Windows" in path:
            return "⚠️ 访问被拒绝：无法读取系统敏感文件。"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 限制返回长度，防止上下文溢出
        if len(content) > 5000:
            return content[:5000] + "\n... (文件过长，已截断)"
        return content
    except FileNotFoundError:
        return f"❌ 文件不存在: {path}"
    except Exception as e:
        return f"❌ 读取失败: {str(e)}"

@server.tool()
async def execute_command(command: str) -> str:
    """
    执行系统 Shell 命令（仅支持只读命令，如 ls、cat、dir）。
    输入参数：command - 要执行的命令字符串。
    返回命令输出结果。
    """
    # 安全黑名单：禁止执行写入/删除/网络请求等危险命令
    dangerous = ["rm", "del", "format", "mkfs", "dd", "wget", "curl", "nc", "telnet"]
    for word in dangerous:
        if word in command.lower().split():
            return f"⚠️ 安全限制：禁止执行包含 '{word}' 的命令。"
    
    try:
        # 执行命令，超时 5 秒
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        output = result.stdout + result.stderr
        if len(output) > 5000:
            return output[:5000] + "\n... (输出过长，已截断)"
        return output or "(命令无输出)"
    except subprocess.TimeoutExpired:
        return "❌ 命令执行超时（5 秒）"
    except Exception as e:
        return f"❌ 命令执行失败: {str(e)}"


if __name__ == "__main__":
    server.run(transport="sse")