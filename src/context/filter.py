from typing import List, Dict, Any, Optional
import re

class ContextFilter:
    def __init__(
            self,
            min_doc_score: float = 0.7,
            max_docs: int = 5,
            max_conversation_turns: int = 10,
            min_message_length: int = 3,
            ):
        self.min_doc_score = min_doc_score
        self.max_docs = max_docs
        self.max_conversation_turns = max_conversation_turns
        self.min_message_length = min_message_length

        self.ignore_patterns = [
            r"^(你好|hi|hello|嗨|您好)$",
            r"^(谢谢|感谢|多谢|thx|thanks)$",
            r"^(再见|拜拜|bye|goodbye)$",
            r"^(好|好的|ok|嗯|哦|知道了)$",
            r"^(能帮我|请问|麻烦|请教).{0,5}[?？]?$",
        ]
        self._ignore_regex = [re.compile(p, re.IGNORECASE) for p in self.ignore_patterns]

    def filter_docs(self, docs: List[Dict]) -> List[Dict]:#过滤 RAG 检索结果
        if not docs:
            return []
        filtered = [doc for doc in docs if doc.get("score", 0) >= self.min_doc_score]
        filtered.sort(key=lambda x: x.get("score", 0), reverse=True)

        if len(filtered) > self.max_docs:
            filtered = filtered[:self.max_docs]
            print(f"📄 Context Filter: 文档从 {len(docs)} 条压缩到 {len(filtered)} 条")
        return filtered

    def filter_messages(self, messages: List[Dict]) -> List[Dict]:#过滤对话历史消息
        if not messages:
            return []
        system_messages = []
        conversation_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                conversation_messages.append(msg)

        meaningful_messages = []
        for msg in conversation_messages:
            content = msg.get("content", "").strip()
            is_ignored = False
            for pattern in self._ignore_regex:
                if pattern.match(content):
                    is_ignored = True
                    break

            if not is_ignored and len(content) >= self.min_message_length:
                meaningful_messages.append(msg)

        max_messages = self.max_conversation_turns * 2
        if len(meaningful_messages) > max_messages:
            meaningful_messages = meaningful_messages[-max_messages:]
            print(f"💬 Context Filter: 对话从 {len(conversation_messages)} 条压缩到 {len(meaningful_messages)} 条")
        return system_messages + meaningful_messages

    def deduplicate_tool_results(self, messages: List[Dict]) -> List[Dict]:#去除重复的工具调用结果
        if not messages:
            return []
        seen_contents = set()
        deduped = []
        for msg in messages:
            if msg.get("role") == "tool" or hasattr(msg, "tool_call_id"):
                content = msg.get("content", "")
                key = content[:200] if content else ""
                if key and key in seen_contents:
                     continue
                if key:
                    seen_contents.add(key)
            deduped.append(msg)

        if len(deduped) < len(messages):
            print(f"🧹 Context Filter: 去除了 {len(messages) - len(deduped)} 条重复工具结果")
        return deduped

    def filter_context(# 一站式过滤
        self,
        messages: List[Dict],
        docs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        original_msg_count = len(messages)
        original_doc_count = len(docs) if docs else 0

        deduped_messages = self.deduplicate_tool_results(messages)# 1. 去重工具结果
        filtered_messages = self.filter_messages(deduped_messages)# 2. 过滤对话
        filtered_docs = self.filter_docs(docs) if docs else [] # 3. 过滤文档

        return {
            "messages": filtered_messages,
            "docs": filtered_docs,
            "stats": {
                "original_message_count": original_msg_count,
                "filtered_message_count": len(filtered_messages),
                "original_doc_count": original_doc_count,
                "filtered_doc_count": len(filtered_docs),
                "messages_reduced": original_msg_count - len(filtered_messages),
                "docs_reduced": original_doc_count - len(filtered_docs),
            }
        }

_global_filter : Optional[ContextFilter]= None

def get_context_filter(
    min_doc_score: float = 0.7,
    max_docs: int = 5,
    max_conversation_turns: int = 10,
) -> ContextFilter:
        global _global_filter
        if _global_filter is None:
            _global_filter = ContextFilter(
            min_doc_score=min_doc_score,
            max_docs=max_docs,
            max_conversation_turns=max_conversation_turns,
        )
        return _global_filter