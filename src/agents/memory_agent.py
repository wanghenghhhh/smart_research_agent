# src/agents/memory_agent.py

from typing import Dict, List
from langchain_openai import ChatOpenAI
from config.settings import settings

class MemoryAgent:
    """
    记忆查询处理器：直接从对话历史中回答问题
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
    
    async def answer(self, query: str, messages: List[Dict]) -> str:
        """
        基于对话历史回答问题
        
        Args:
            query: 用户当前问题
            messages: 历史消息列表（LangGraph 的 messages 字段）
        
        Returns:
            回答文本（直接展示给用户）
        """
        # 1. 提取历史提问（只取 user 角色的消息）
        history_questions = []
        for msg in messages:
            if msg.get("role") == "user":
                history_questions.append(msg.get("content", ""))
        
        # 2. 构建上下文，供 LLM 生成回答
        history_text = "\n".join([
            f"第{i+1}轮: {q}" for i, q in enumerate(history_questions)
        ]) if history_questions else "（无历史记录）"
        
        # 3. 用 LLM 生成友好回答
        prompt = f"""
你是一个智能助手。用户问: "{query}"

以下是用户的对话历史:
{history_text}

请根据用户的问题和对话历史，给出简洁、准确的回答。
- 如果是问"我之前问过什么"，直接列出历史问题（最多5条）
- 如果是问"刚才说了什么"，引用最近一条对话
- 如果历史为空，告诉用户"目前还没有历史记录"
- 回答要亲切、自然，不要生成报告格式
"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content