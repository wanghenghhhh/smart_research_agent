from typing import Dict, Literal
from langchain_openai import ChatOpenAI
from config.settings import settings

class IntentRouter:
    """
    意图分流器：判断用户输入是"研究任务"还是"对话/记忆查询"
    """
    
    def __init__(self):
        # 1. 用轻量模型判断意图（温度设为0，确保确定性）
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
    
    def route(self, query: str) -> Literal["memory_query", "chat", "research"]:
        """
        根据用户输入判断意图类型
        """
        clean_query = query.strip().lower()
        
        # ✅ 第一层：严格过滤极短的纯打招呼/无意义闲聊（防止误伤学术问题）
        strict_chat_words = {"你好", "您好", "hi", "hello", "谢谢", "感谢", "再见", "拜拜", "bye", "help", "帮助"}
        if clean_query in strict_chat_words:
            return "chat"
            
        # 显式记忆查询关键词
        memory_keywords = ["我之前", "我们之前", "刚才我说", "你记得", "之前聊过", "上次说的", "以前问过"]
        for kw in memory_keywords:
            if kw in clean_query:
                return "memory_query"

        # ✅ 第二层：交由 LLM 准确判定（针对复杂的学术/技术/科研问题）
        return self._llm_intent(query)
    
    def _llm_intent(self, query: str) -> str:
        """使用 LLM 强约束 Prompt 判断意图"""
        prompt = f"""你是一个智能路由分类器。请评估用户输入的意图，并严格返回指定单词之一：[research, chat, memory_query]。

分类标准：
1. research（核心分类）：
   - 任何涉及技术概念解释、行业分析、知识解答、代码开发、专业领域定义的问题（如：“什么是大模型”、“大模型开发需要什么”、“如何搭建RAG”等）。
   - 只要用户在询问某个知识点或寻求信息解答，一律分类为 research。

2. chat：
   - 纯粹的打招呼、表达感谢、无意义闲聊、对 Agent 本身身份的询问（如：“你是谁”、“你能做什么”）。

3. memory_query：
   - 用户明确追问先前的对话历史或个人偏好记忆（如：“我刚才说了什么”、“你还记得我的名字吗”）。

用户输入: {query}

请只输出一个分类单词（research / chat / memory_query），不要包含任何其他标点或说明文本：
"""
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip().lower()
            
            # 过滤提取可能的单词
            for candidate in ["research", "chat", "memory_query"]:
                if candidate in result:
                    return candidate
            return "research"  # 默认降级为研究模式
        except Exception as e:
            print(f"⚠️ 意图识别 LLM 调用异常: {e}，默认降级为 research")
            return "research"  # 异常时默认走研究流程，保障业务不中断