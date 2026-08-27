from typing import Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.tools.definitions import TOOLS as LOCAL_TOOLS
from config.settings import settings
from src.mcp.client import get_mcp_client, create_mcp_tool_adapter

class ReActAgent:
    """
    ReAct 模式的自主决策 Agent
    它自己决定：调用哪个工具、何时结束
    """
    
    def __init__(self,use_mcp:bool = True):
        self.use_mcp = use_mcp
        self.mcp_client = None
        self.tools = []
        self._initialize_tools()
        # 1. 初始化 LLM 并绑定工具
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        ).bind_tools(self.tools)
        
        self.tools_node = ToolNode(self.tools)
        
        # 3. 构建图
        self.graph = StateGraph(MessagesState)
        self.graph.add_node("agent", self._call_model)
        self.graph.add_node("tools", self.tools_node)
        
        # 入口
        self.graph.set_entry_point("agent")
        
        # 条件边：如果 LLM 调用工具 → 去 tools；否则结束
        self.graph.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # 工具执行完后回到 agent 继续思考
        self.graph.add_edge("tools", "agent")
        
        # 4. 编译图（不传 recursion_limit）
        self.runnable = self.graph.compile()
    def _initialize_tools(self):
        """初始化工具：优先 MCP，回退本地"""
        self.tools = LOCAL_TOOLS.copy()
        if self.use_mcp:
            try:
                print("🔌 尝试连接 MCP Server...")
                self._mcp_ready = False
                self._mcp_tools = []
                print("⚠️ MCP 工具将在首次运行时加载")
            except Exception as e:
                print(f"⚠️ MCP 初始化失败: {e}，降级使用本地工具")

    
    async def _ensure_mcp_tools(self):
        """确保 MCP 工具已加载（延迟加载）"""
        if not self.use_mcp or hasattr(self, '_mcp_loaded'):
            return
        
        try:
            from src.mcp.client import get_mcp_client, create_mcp_tool_adapter
            
            # ✅ 使用全局长连接单例，不要用 async with（避免执行完立刻被 close 断开）
            mcp_client = await get_mcp_client()
            mcp_tool_infos = await mcp_client.list_tools()
            print(f"🔌 从 MCP Server 发现 {len(mcp_tool_infos)} 个工具")
            
            mcp_tools = []
            for info in mcp_tool_infos:
                adapter = create_mcp_tool_adapter(mcp_client, info)
                mcp_tools.append(adapter)
            
            mcp_names = {t.name for t in mcp_tools}
            local_filtered = [t for t in LOCAL_TOOLS if t.name not in mcp_names]
            self.tools = mcp_tools + local_filtered
            
            self.llm = self.llm.bind_tools(self.tools)
            self.tools_node = ToolNode(self.tools)
            
            from langgraph.graph import StateGraph, END, MessagesState
            self.graph = StateGraph(MessagesState)
            self.graph.add_node("agent", self._call_model)
            self.graph.add_node("tools", self.tools_node)
            self.graph.set_entry_point("agent")
            self.graph.add_conditional_edges(
                "agent",
                tools_condition,
                {"tools": "tools", END: END}
            )
            self.graph.add_edge("tools", "agent")
            self.runnable = self.graph.compile()
            
            self._mcp_loaded = True
            print(f"✅ MCP 工具加载成功: {[t.name for t in mcp_tools]}")
            
        except Exception as e:
            print(f"⚠️ MCP 工具加载失败: {e}，继续使用本地工具")
            self._mcp_loaded = True
    
    async def _call_model(self, state: MessagesState) -> Dict:
        """调用 LLM 决定下一步"""
        messages = state["messages"]
        response = await self.llm.ainvoke(messages)
        return {"messages": [response]}
    
    # src/agents/react_agent.py

    async def run(self, task_content: str, context: Dict = None) -> str:
        await self._ensure_mcp_tools()

        history_messages = context.get("messages", []) if context else []

        if history_messages:
            from src.context.pipeline import get_context_pipeline

            pipeline = get_context_pipeline(
            max_context_tokens=4000,
            model=settings.OPENAI_MODEL,
        )
            processed = await pipeline.process(
            messages=history_messages,
            docs=[],  # 当前没有单独传 docs，docs 已经在消息中
        )
            processed_messages = processed["messages"]
            print(f"📦 [ReActAgent] 上下文已优化: {len(history_messages)} -> {len(processed_messages)} 条消息")
        else:
            processed_messages = []
        
        # 判断是否是操作型任务（包含工具调用关键词）
        # 让 Agent 直接回答，不经过 Reporter
        operation_keywords = ["读取", "查看", "执行", "列出", "显示"]
        
        # 构建提示词
        if any(keyword in task_content for keyword in operation_keywords):
            # 操作型任务：直接展示结果，不要报告
            instruction = """
    你拥有 `read_file`、`execute_command` 等工具。

    ⚠️ **规则**：
    1. 调用工具后，把工具返回的内容**原样展示**在最终答案里。
    2. 用 ``` 代码块包裹内容。
    3. **禁止总结、禁止解释、禁止生成报告**。
    4. 展示完内容后立即停止。

    直接开始执行。
    """
        else:
            # 研究型任务：正常搜索，后续由 Reporter 处理
            instruction = """
    你是科研助手。调用 search_web 或 search_knowledge_base 搜索信息。
    收集到足够信息后，用中文输出答案。
    """

        initial_messages = [
            HumanMessage(content=f"""
    你的任务: {task_content}

    {instruction}

    可用工具:
    - read_file: 读取本地文件
    - execute_command: 执行系统命令
    - search_web: 搜索网络
    - search_knowledge_base: 搜索知识库

    开始执行。
    """)
        ]
        
        config = {"recursion_limit": 10}
        final_state = await self.runnable.ainvoke(
            {"messages": initial_messages},
            config=config
        )
        
        # 提取最终回答
        messages = final_state["messages"]
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                # ✅ 如果是操作型任务，直接把答案存入 context，让 main.py 展示
                if any(keyword in task_content for keyword in operation_keywords):
                    if context is not None:
                        context["final_report"] = msg.content
                        context["research_complete"] = True
                    return msg.content
                return msg.content
        
        return "Agent 未能生成答案。"