from typing import Dict, List,AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from datetime import datetime

class Reporter:
    """
    报告生成器：基于所有研究结果生成Markdown格式的研究报告。
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,  # 低温度使输出更确定、更正式
            api_key=settings.OPENAI_API_KEY
        )
        
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的科研报告撰写专家。基于研究计划和搜索结果，生成一份结构清晰、内容详实的Markdown格式研究报告。

报告格式要求：
1. 使用标准Markdown格式
2. 包含：标题、摘要、引言、方法、结果、讨论、结论、参考文献
3. 结果部分要有数据支撑（如搜索结果中的发现）
4. 引用来源要标注URL

请生成一份专业、完整的研究报告。
"""),
            ("human", """
## 研究问题
{query}

## 研究计划
{plan}

## 搜索发现
{findings}

## 任务执行记录
{history}

请基于以上信息生成研究报告。
""")
        ])
    
    async def generate(self, state: Dict) -> Dict:
        if state.get("final_report"):
            return state
        """
        生成最终报告
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态，包含 final_report
        """
        try:
            query = state.get("query", "")
            plan = state.get("research_plan", "无详细计划")
            findings = self._format_findings(state.get("search_results", []))
            history = self._format_history(state.get("task_history", []))
            
            # 调用LLM生成报告内容
            response = await self.llm.ainvoke(
                self.report_prompt.format(
                    query=query,
                    plan=plan,
                    findings=findings,
                    history=history
                )
            )
            
            report_content = response.content
            
            # 添加元数据和生成时间
            metadata = self._generate_metadata(state)
            final_report = f"""{metadata}

{report_content}

---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*基于 {len(state.get("search_results", []))} 条搜索结果生成*
"""
            
            state["final_report"] = final_report
            state["research_complete"] = True
            
            return state
            
        except Exception as e:
            state["error"] = f"报告生成失败: {str(e)}"
            return state

    async def generate_stream(self, state: Dict) -> AsyncGenerator[str, None]:
        query = state.get("query", "")
        plan = state.get("research_plan", "无详细计划")
        findings = self._format_findings(state.get("search_results", []))
        history = self._format_history(state.get("task_history", []))
        metadata = f"""---
        title: "研究报告"
        query: "{query}"
        total_tasks: {len(state.get('tasks', []))}
        completed_tasks: {len(state.get('task_history', []))}
        ---
        """
        yield metadata + "\n\n"

        prompt_value = self.report_prompt.format(
        query=query,
        plan=plan,
        findings=findings,
        history=history
    )
        async for chunk in self.llm.astream(prompt_value):
            content = chunk.content
            if content:
                yield content
        footer = f"""
        ---
        *报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
        *基于 {len(state.get('search_results', []))} 条搜索结果生成*
        """
        yield footer

    def _format_findings(self, search_results: List[Dict]) -> str:
        """
        将搜索结果格式化为可读的文本块
        """
        if not search_results:
            return "⚠️ 未通过工具收集到有效信息。Agent 可能未能成功调用搜索工具，或工具返回结果为空。建议检查网络连接或搜索关键词。"
        
        formatted = []
        for i, result in enumerate(search_results[:10], 1):  # 最多取10条
            content = result.get('content','')
            if "标题" in content:
                formatted.append(f"### 发现 {i}\n{content}")
            else:
                formatted.append(f"### 发现 {i}\n- 内容: {content[:300]}...")
        
        return "\n".join(formatted)
    
    def _format_history(self, task_history: List[Dict]) -> str:
        """
        格式化任务执行历史
        """
        if not task_history:
            return "无任务执行记录。"
        
        formatted = []
        for task in task_history:
            formatted.append(f"""
- 任务 {task.get('task_id')}: {task.get('content', '')}
  类型: {task.get('task_type', '')}
  状态: {task.get('status', '')}
""")
        return "\n".join(formatted)
    
    def _generate_metadata(self, state: Dict) -> str:
        """
        生成报告开头的YAML式元数据
        """
        return f"""---
title: "研究报告"
query: "{state.get('query', '')}"
total_tasks: {len(state.get('tasks', []))}
completed_tasks: {len(state.get('task_history', []))}
total_search_results: {len(state.get('search_results', []))}
---"""