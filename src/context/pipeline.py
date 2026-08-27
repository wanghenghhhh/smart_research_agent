from typing import List, Dict, Any, Optional, Tuple
from src.context.filter import get_context_filter
from src.context.compressor import get_context_compressor
from src.context.budget import get_budget_manager
from src.utils.token_counter import count_messages_tokens

class ContextPipeline:#串联执行：过滤 → 压缩 → 预算管理
    def __init__(
        self,
        min_doc_score: float = 0.7,
        max_docs: int = 5,
        max_conversation_turns: int = 10,
        max_context_tokens: int = 4000,
        model: str = "gpt-4",
    ):
        self.filter = get_context_filter(
            min_doc_score=min_doc_score,
            max_docs=max_docs,
            max_conversation_turns=max_conversation_turns,
        )

        self.compressor = get_context_compressor(
            max_context_tokens=max_context_tokens,
        )

        self.budget_manager = get_budget_manager(
            total_budget=max_context_tokens,
            model=model,
        )
        self.model = model
        self.max_context_tokens = max_context_tokens

    async def process(
        self,
        messages: List[Dict],
        docs: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        stats = {
            "original_message_count": len(messages),
            "original_doc_count": len(docs) if docs else 0,
            "filter_stage": {},
            "compress_stage": {},
            "budget_stage": {},
        }
        print("\n" + "=" * 50)
        print("🔧 [Context Pipeline] 开始处理上下文...")
        print("=" * 50)
        print("\n📌 Stage 1: 上下文过滤")
        filter_result = self.filter.filter_context(messages, docs)
        filtered_messages = filter_result["messages"]
        filtered_docs = filter_result["docs"]
        stats["filter_stage"] = filter_result["stats"]
        print(f"   消息: {len(messages)} -> {len(filtered_messages)}")
        print(f"   文档: {len(docs) if docs else 0} -> {len(filtered_docs)}")

        print("\n📌 Stage 2: 上下文压缩")
        compress_result = await self.compressor.compress_context(
            filtered_messages,
            filtered_docs
        )
        compressed_messages = compress_result["messages"]
        compressed_docs = compress_result["docs"]
        stats["compress_stage"] = compress_result["stats"]
        print(f"   消息: {len(filtered_messages)} -> {len(compressed_messages)}")
        print(f"   文档: {len(filtered_docs)} -> {len(compressed_docs)}")

        print("\n📌 Stage 3: Token 预算管理")
        components = self._build_components(compressed_messages, compressed_docs)
        trimmed_components, budget_report = self.budget_manager.fit_into_budget(
            components
        )
        final_messages = self._extract_messages(trimmed_components)
        final_docs = self._extract_docs(trimmed_components)
        stats["budget_stage"] = {
            "total_tokens": budget_report["total_tokens"],
            "budget_limit": budget_report["budget_limit"],
            "usage_percentage": budget_report["usage_percentage"],
            "status": budget_report["status"],
        }
        print(f"   Token: {budget_report['total_tokens']} / {budget_report['budget_limit']}")
        print(f"   状态: {budget_report['status']}")
        print("\n" + "=" * 50)
        print("✅ [Context Pipeline] 处理完成")
        print("=" * 50 + "\n")

        return {
            "messages": final_messages,
            "docs": final_docs,
            "compression_stats": stats,
            "budget_report": budget_report,
        }

    def _build_components(
        self,
        messages: List[Dict],
        docs: List[Dict]
    ) -> Dict[str, Any]:
        components = {
            "system": "",
            "recent_history": [],
            "rag_docs": [],
            "tool_results": [],
            "history_summary": "",
        }
        system_msgs = []
        other_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)
        if system_msgs:
            components["system"] = "\n".join([m.get("content", "") for m in system_msgs])

        components["recent_history"] = other_msgs
        components["rag_docs"] = docs
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        components["tool_results"] = tool_msgs
        for msg in system_msgs:
            content = msg.get("content", "")
            if "历史对话摘要" in content or "[历史对话摘要]" in content:
                components["history_summary"] = content
                break
        return components

    def _extract_messages(self, components: Dict[str, Any]) -> List[Dict]:
        messages = []
        if components.get("system"):
            messages.append({
                "role": "system",
                "content": components["system"]
            })
        if components.get("history_summary"):
            if not components.get("system") or components["history_summary"] not in components["system"]:
                messages.append({
                    "role": "system",
                    "content": components["history_summary"]
                })
        messages.extend(components.get("recent_history", []))
        return messages
    def _extract_docs(self, components: Dict[str, Any]) -> List[Dict]:
        return components.get("rag_docs", [])

_global_pipeline:Optional[ContextPipeline] = None

def get_context_pipeline(
    min_doc_score: float = 0.7,
    max_docs: int = 5,
    max_conversation_turns: int = 10,
    max_context_tokens: int = 4000,
    model: str = "gpt-4",
) -> ContextPipeline:
    """获取全局上下文管道实例"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = ContextPipeline(
            min_doc_score=min_doc_score,
            max_docs=max_docs,
            max_conversation_turns=max_conversation_turns,
            max_context_tokens=max_context_tokens,
            model=model,
        )
    return _global_pipeline