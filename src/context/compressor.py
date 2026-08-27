from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.token_counter import count_messages_tokens, get_encoding
from config.settings import settings

class ContextCompressor:
    def __init__(
        self,
        max_context_tokens: int = 4000,
        max_doc_chars: int = 2000,
        max_tool_chars: int = 1500,
        keep_recent_turns: int = 3,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_doc_chars = max_doc_chars
        self.max_tool_chars = max_tool_chars
        self.keep_recent_turns = keep_recent_turns

        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None
        )

        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个对话摘要专家。请将下面的历史对话压缩成一段简洁的摘要（200字以内）。

要求：
1. 保留关键主题、问题和结论
2. 按时间顺序组织（先发生什么，后发生什么）
3. 不要添加新信息，只压缩已有的内容
4. 不要使用 Markdown 格式

历史对话："""),
            ("human", "{history}")
        ])

    def should_compress(self, messages: List[Dict]) -> bool:
        if not messages:
            return False
        token_count = count_messages_tokens(messages)
        return token_count > self.max_context_tokens

    async def compress_messages(self, messages: List[Dict]) -> List[Dict]:#压缩历史对话
        if not messages:
            return []
        if not self.should_compress(messages):
            print("📦 [Compressor] 上下文未超限，跳过压缩")
            return messages
        print(f"📦 [Compressor] 触发压缩，原始消息数: {len(messages)}")

        system_msgs = []
        conversation_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conversation_msgs.append(msg)
        if len(conversation_msgs) <= self.keep_recent_turns * 2:
            return messages

        keep_count = self.keep_recent_turns * 2
        recent_messages = conversation_msgs[-keep_count:] if len(conversation_msgs) > keep_count else conversation_msgs
        old_messages = conversation_msgs[:-len(recent_messages)] if len(conversation_msgs) > keep_count else []

        if old_messages:
            summary = await self._generate_summary(old_messages)
            summary_msg = {
                "role": "system",
                "content": f"[历史对话摘要] {summary}\n\n注意：以下是最新的对话内容。"
            }
            compressed = system_msgs + [summary_msg] + recent_messages
        else:
            compressed = system_msgs + recent_messages
        print(f"📦 [Compressor] 压缩完成: {len(messages)} -> {len(compressed)} 条消息")
        return compressed

    async def _generate_summary(self, messages: List[Dict]) -> str:#调用 LLM 生成历史对话摘要
        history_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"用户: {content}\n"
            elif role == "assistant":
                history_text += f"助手: {content}\n"
            elif role == "tool":
                history_text += f"工具返回: {content[:200]}...\n"

        if len(history_text) > 3000:
            history_text = history_text[:3000] + "\n... (历史过长，已截断)"
            try:
                response = await self.llm.ainvoke(
                self.summary_prompt.format(history=history_text)
            )
                return response.content.strip()
            except Exception as e:
                print("⚠️ [Compressor] 摘要生成失败: {e}")
                return history_text[:500] + "..." if len(history_text) > 500 else history_text

    def truncate_docs(self, docs: List[Dict]) -> List[Dict]:#文档截取
        if not docs:
            return []
        truncated = []
        for doc in docs:
            content = doc.get("content", "")
            if len(content) > self.max_doc_chars:
                doc["content"] = content[:self.max_doc_chars] + "\n... (内容过长，已截断)"
                truncated.append(doc)
            else:
                truncated.append(doc)
        if len(truncated) != len(docs):
            print(f"📄 [Compressor] 截断了 {len(docs) - len(truncated)} 篇过长文档")
        return truncated

    def truncate_tool_results(self, messages: List[Dict]) -> List[Dict]:#工具截取
        if not messages:
            return []
        truncated_messages = []
        for msg in messages:
            if msg.get("role") == "tool" or hasattr(msg, "tool_call_id"):
                content = msg.get("content", "")
                if len(content) > self.max_tool_chars:
                    msg["content"] = content[:self.max_tool_chars] + "\n... (输出过长，已截断)"
            truncated_messages.append(msg)
        return truncated_messages

    async def compress_context(#一站式压缩
        self,
        messages: List[Dict],
        docs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        original_msg_count = len(messages)
        original_doc_count = len(docs) if docs else 0

        tool_truncated = self.truncate_tool_results(messages)
        compressed_messages = await self.compress_messages(tool_truncated)
        truncated_docs = self.truncate_docs(docs) if docs else []

        return {
            "messages": compressed_messages,
            "docs": truncated_docs,
            "stats": {
                "original_message_count": original_msg_count,
                "compressed_message_count": len(compressed_messages),
                "original_doc_count": original_doc_count,
                "truncated_doc_count": len(truncated_docs),
                "compression_ratio": f"{len(compressed_messages)}/{original_msg_count}",
            }
        }

_global_compressor : Optional[ContextCompressor]= None

def get_context_compressor(
    max_context_tokens: int = 4000,
    max_doc_chars: int = 2000,
    max_tool_chars: int = 1500,
) -> ContextCompressor:
    global _global_compressor
    if _global_compressor is None:
        _global_compressor = ContextCompressor(
            max_context_tokens=max_context_tokens,
            max_doc_chars=max_doc_chars,
            max_tool_chars=max_tool_chars,
        )
    return _global_compressor