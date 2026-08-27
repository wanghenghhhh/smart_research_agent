from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
import re

class Planner:
    """
    研究规划器：将用户问题转化为结构化的研究任务列表。
    使用LLM生成计划，然后解析出任务。
    """
    
    def __init__(self):
        # 初始化OpenAI聊天模型
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )
        
        # 定义规划提示词模板
        self.plan_prompt = ChatPromptTemplate.from_messages( [
            ("system", """你是一个科研研究规划专家。你的任务是将用户的研究问题分解为具体、可执行的研究任务。

请按照以下格式输出研究计划：

## 研究计划概述
[简要描述整体研究思路]

## 具体研究任务
1. [任务1描述] | 类型: [search/read/analyze]
2. [任务2描述] | 类型: [search/read/analyze]
3. [任务3描述] | 类型: [search/read/analyze]
...

注意：
- 每个任务要具体、可执行
- 任务类型必须是：search（搜索）、read（阅读）、analyze（分析）
- 通常3-5个任务即可
- 任务之间要有逻辑递进关系
"""),
            ("human", "用户的研究问题是：{query}")
        ])
    
    async def plan(self, state: Dict) -> Dict:
        """
        执行规划，更新状态中的 research_plan 和 tasks
        
        参数:
            state: 当前状态字典
            
        返回:
            更新后的状态字典
        """
        query = state.get("query", "")
        
        if not query:
            state["error"] = "研究问题不能为空"
            return state
        memory_keywords = ["我之前", "我们之前", "刚才", "刚才我说", "你记得", "之前聊过", "上次"]
        is_memory_query = any(kw in query for kw in memory_keywords)
        if is_memory_query:
            state["research_plan"] = "记忆查询：从对话历史中检索信息"
            state["tasks"] = [{
                "id": 1,
                "content": query,
                "type": "memory_query",  # 特殊类型
                "status": "pending"
            }]
            state["current_task_index"] = 0
            state["research_complete"] = False
            state["search_results"] = []
            state["task_history"] = []
            state["is_memory_query"] = True 
            return state

        try:
            # 调用LLM生成计划
            response = await self.llm.ainvoke(
                self.plan_prompt.format(query=query)
            )
            
            plan_content = response.content
            state["research_plan"] = plan_content
            
            # 解析任务列表
            tasks = self._parse_tasks(plan_content)
            state["tasks"] = tasks
            state["current_task_index"] = 0
            state["research_complete"] = False
            state["search_results"] = []
            state["task_history"] = []
            
            return state
            
        except Exception as e:
            state["error"] = f"规划失败: {str(e)}"
            return state
    
    def _parse_tasks(self, plan_text: str) -> List[Dict]:
        """
        从计划文本中解析出任务列表
        
        参数:
            plan_text: LLM生成的计划文本
            
        返回:
            任务列表，每个任务包含 id, type, content, status
        """
        tasks = []
        
        # 正则匹配格式: "1. 任务描述 | 类型: search"
        pattern = r'(\d+)\.\s*(.+?)\s*\|\s*类型:\s*(\w+)'
        matches = re.findall(pattern, plan_text)
        
        if matches:
            for idx, (num, content, task_type) in enumerate(matches):
                tasks.append({
                    "id": idx + 1,
                    "content": content.strip(),
                    "type": task_type.strip().lower(),
                    "status": "pending"  # pending, running, completed, failed
                })
        else:
            # 如果正则匹配失败，尝试按行分割（容错）
            lines = plan_text.split('\n')
            task_count = 0
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # 去掉序号前缀
                    if line[0].isdigit() and '.' in line:
                        content = line.split('.', 1)[1].strip()
                    elif line.startswith('-') or line.startswith('•'):
                        content = line[1:].strip()
                    else:
                        content = line
                    
                    # 根据关键词猜测任务类型
                    task_type = "search"
                    if "分析" in content or "总结" in content:
                        task_type = "analyze"
                    elif "阅读" in content or "查看" in content:
                        task_type = "read"
                    
                    task_count += 1
                    tasks.append({
                        "id": task_count,
                        "content": content,
                        "type": task_type,
                        "status": "pending"
                    })
        
        # 如果仍然没有解析到任何任务，创建一个默认任务（兜底）
        if not tasks:
            tasks.append({
                "id": 1,
                "content": f"全面研究: {plan_text[:100]}...",
                "type": "search",
                "status": "pending"
            })
        
        return tasks