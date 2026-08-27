from typing import Dict, Literal,AsyncGenerator,Any
from langgraph.graph import StateGraph, END
import sqlite3,os
from src.core.state import ResearchState
from src.agents.intent_router import IntentRouter
from src.agents.memory_agent import MemoryAgent
from src.agents.planner import Planner
from src.agents.react_agent import ReActAgent
from src.agents.reporter import Reporter
from src.evaluation import get_tracer,MetricsCollector

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    HAS_SQLITE = True
except ImportError:
    # 降级方案：如果没有安装，使用内存模式（但会提示）
    HAS_SQLITE = False
    print("⚠️ langgraph-checkpoint-sqlite 未安装，将使用内存模式（重启丢失记忆）")
    SqliteSaver = None

class ResearchWorkflow:
    """
    MVP Agent工作流
    流程: START → Planner → Executor (循环) → Reporter → END
    """
    
    def __init__(self,db_path: str = "checkpoints.db", use_memory: bool = True):
        # 初始化各个节点
        self.intent_router = IntentRouter()
        self.memory_agent = MemoryAgent()
        self.planner = Planner()
        self.react_agent = ReActAgent()
        self.reporter = Reporter()

        self.db_path = db_path
        self.use_memory = use_memory and HAS_SQLITE

        self._build_graph()
    
    def _build_graph(self):
        """构建工作流图"""
        # 创建状态图，传入状态类型
        self.graph = StateGraph(ResearchState)
        
        # 添加节点，每个节点是一个异步函数
        self.graph.add_node("intent_router", self._route_node)
        self.graph.add_node("memory_query", self._memory_node)
        self.graph.add_node("chat", self._chat_node)
        self.graph.add_node("planner", self._plan_node)
        self.graph.add_node("react_agent", self._react_node)
        self.graph.add_node("reporter", self._report_node)
        
        # 设置入口点
        self.graph.set_entry_point("intent_router")
        self.graph.add_conditional_edges(
            "intent_router",
            self._route_after_intent,
            {
                "memory_query": "memory_query",
                "chat": "chat",
                "research": "planner",
                "end": END
            }
        )
        self.graph.add_edge("memory_query", END)
        self.graph.add_edge("chat", END)

        self.graph.add_edge("planner", "react_agent")
        self.graph.add_conditional_edges(
            "react_agent",
            self._should_continue,
            {
                "continue": "react_agent",  
                "finish": "reporter",  
                "end": END,     
                "error": END
            }
        )
        self.graph.add_edge("reporter", END)
    async def _route_node(self, state: Dict) -> Dict:
        """意图分流节点"""
        query = state.get("query", "")
        intent = self.intent_router.route(query)
        state["intent_type"] = intent
        print(f"🧠 意图识别: {intent}")
        return state
    def _route_after_intent(self, state: Dict) -> str:
        """根据意图选择路径"""
        intent = state.get("intent_type", "research")
        if state.get("error"):
            return "end"
        return intent
    async def _memory_node(self, state: Dict) -> Dict:
        """记忆查询节点"""
        query = state.get("query", "")
        messages = state.get("messages", [])
        answer = await self.memory_agent.answer(query, messages)
        state["final_report"] = answer
        state["is_memory_query"] = True
        state["research_complete"] = True
        return state
    async def _chat_node(self, state: Dict) -> Dict:
        """聊天节点"""
        query = state.get("query", "")
        # 简单聊天回复
        if "你好" in query or "hi" in query.lower():
            state["final_report"] = "你好！我是智能科研助手。我可以帮你：\n1. 研究科研问题\n2. 查询知识库\n3. 读取文件\n输入问题开始吧！"
        elif "谢谢" in query:
            state["final_report"] = "不客气！有需要随时找我。"
        else:
            state["final_report"] = "我是科研助手，可以帮你研究学术问题。请问有什么可以帮你的？"
        state["research_complete"] = True
        return state

    
    async def _plan_node(self, state: Dict) -> Dict:
        """规划节点的包装函数"""
        return await self.planner.plan(state)
    async def _react_node(self, state: Dict) -> Dict:
        """
        Agent 决策节点：执行当前任务
        """
        task_index = state.get("current_task_index", 0)
        tasks = state.get("tasks", [])
        
        if task_index >= len(tasks):
            state["research_complete"] = True
            return state
        
        current_task = tasks[task_index]
        task_content = current_task.get("content", "")

        if state.get("final_report") is not None:
            state["research_complete"] = True
            return state
        
        # 让 ReAct Agent 自主完成这个任务
        result = await self.react_agent.run(task_content, context=state)
        
        # 记录结果
        state["task_history"].append({
            "task_id": current_task["id"],
            "content": task_content,
            "result": result,
            "status": "completed"
        })
        
        state["current_task_index"] = task_index + 1
        
        if state["current_task_index"] >= len(tasks):
            state["research_complete"] = True
        
        return state
    
    async def _report_node(self, state: Dict) -> Dict:
        """报告节点的包装函数"""
        if state.get("final_report"):
            return state
        return await self.reporter.generate(state)
    
    def _should_continue(self, state: Dict) -> Literal["continue", "finish", "end","error"]:
        """
        判断执行流程的下一步
        
        返回:
            "continue": 继续执行下一个任务
            "finish": 所有任务完成，生成报告
            "error": 发生错误，终止
        """
        if state.get("error"):
            return "error"
        if state.get("final_report") is not None:
            return "end"
        if state.get("research_complete", False):
            return "finish"
        
        # 检查是否还有未执行的任务
        tasks = state.get("tasks", [])
        current_index = state.get("current_task_index", 0)
        if current_index < len(tasks):
            return "continue"
        else:
            # 如果没有任务但 research_complete 为 False，手动设为完成
            state["research_complete"] = True
            return "finish"
    
    async def run(self, query: str, thread_id: str = "default") -> Dict:
        tracer = get_tracer()
        tracer.start_trace(f"Research: {query[:30]}...")
        # 初始化状态
        initial_state: ResearchState = {
            "query": query,
            "research_plan": None,
            "tasks": [],
            "current_task_index": 0,
            "task_history": [],
            "search_results": [],
            "retrieved_docs": [],
            "research_complete": False,
            "final_report": None,
            "error": None,
            "intermediate_steps": [],
            "query_complexity": None,
            "adaptive_k": None,
            "retrieval_source": "none",
            "is_action_task": False
        }

        config = {"configurable": {"thread_id": thread_id}}
        tracer.start_span("workflow_execution", "root", metadata={"query": query})

        try:
            if self.use_memory:
                async with AsyncSqliteSaver.from_conn_string(self.db_path) as checkpointer:
                    app = self.graph.compile(checkpointer=checkpointer)
                    final_state = await app.ainvoke(initial_state, config=config)
            else:
                app = self.graph.compile()
                final_state = await app.ainvoke(initial_state, config=config)

            # ✅ 结束 workflow_execution 跨度
            final_report_preview = final_state.get("final_report", "")
            if final_report_preview:
                preview = final_report_preview[:200] + "..." if len(final_report_preview) > 200 else final_report_preview
            else:
                preview = "(无输出)"

            metrics = MetricsCollector()
            metrics.collect_from_trace(tracer.get_report())
            metrics.print_summary()
            tracer.end_span(output_data=preview)
            tracer.end_trace()

            return final_state

        except Exception as e:
        # ✅ 出错时也要结束追踪
            tracer.end_span(output_data=f"❌ Error: {str(e)}")
            tracer.end_trace()
            raise
    async def run_stream(self, query: str, thread_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
        from src.evaluation.tracer import get_tracer
        yield {"type": "status", "content": "🔍 正在分析问题..."}
        yield {"type": "status", "content": "📋 正在制定研究计划..."}
        initial_state = {
        "query": query,
        "messages": [],
        "research_plan": None,
        "tasks": [],
        "current_task_index": 0,
        "task_history": [],
        "search_results": [],
        "retrieved_docs": [],
        "research_complete": False,
        "final_report": None,
        "error": None,
        "intermediate_steps": [],
        "query_complexity": None,
        "adaptive_k": None,
        "retrieval_source": "none",
        "is_action_task": False
    }
        state = await self._plan_node(initial_state)
        if state.get("error"):
            yield {"type": "error", "content": state["error"]}
            return
        tasks = state.get("tasks", [])
        yield {"type": "status", "content": f"✅ 研究计划已生成，共 {len(tasks)} 个任务"}

        while not state.get("research_complete", False):
            task_index = state.get("current_task_index", 0)
            
            if task_index >= len(tasks):
                break
            
            current_task = tasks[task_index]
            task_content = current_task.get("content", "")
            
            yield {
                "type": "status",
                "content": f"🔹 执行任务 {task_index+1}/{len(tasks)}: {task_content[:50]}..."
            }
            
            # 执行当前任务
            state = await self._react_node(state)
            
            # 检查是否有工具调用结果
            task_history = state.get("task_history", [])
            if task_history:
                last_result = task_history[-1]
                result_content = last_result.get("result", "")
                if result_content:
                    yield {
                        "type": "tool_result",
                        "content": str(result_content)[:200] + ("..." if len(str(result_content)) > 200 else "")
                    }
        if state.get("final_report"):
            yield {"type": "answer", "content": state["final_report"]}
            yield {"type": "done", "content": "✅ 完成"}
            return
        yield {"type": "status", "content": "📝 正在生成最终报告..."}
        if hasattr(self.reporter, 'generate_stream'):
            report_chunks = []
            async for chunk in self.reporter.generate_stream(state):
                report_chunks.append(chunk)
                yield {"type": "answer_chunk", "content": chunk}
            
            full_report = "".join(report_chunks)
            state["final_report"] = full_report
            state["research_complete"] = True
        else:
            # 降级方案：使用同步 generate
            state = await self._report_node(state)
            if state.get("final_report"):
                yield {"type": "answer", "content": state["final_report"]}
        
        yield {"type": "done", "content": "✅ 研究完成"}

    def list_sessions(self) -> list:
        """列出所有历史会话 ID"""
        if not self.use_memory or not os.path.exists(self.db_path):
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY checkpoint_id DESC")
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception:
            return []

    def delete_session(self, thread_id: str):
        """删除指定会话的所有数据"""
        if not self.use_memory or not os.path.exists(self.db_path):
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
            conn.close()
            print(f"🗑️ 已删除会话: {thread_id}")
        except Exception as e:
            print(f"⚠️ 删除失败: {e}")

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("🔌 SQLite 连接已关闭")