from typing import List, Dict, Any, Optional, Tuple
from src.utils.token_counter import count_tokens, count_messages_tokens, get_encoding

class TokenBudgetManager:
    def __init__(
        self,
        total_budget: int = 4000,
        model: str = "gpt-4",
    ):
        self.total_budget = total_budget
        self.model = model
        self.encoding = get_encoding(model)

        self.priority_order = [
            "system",          # 系统提示词（最高优先级）
            "recent_history",  # 最近 2-3 轮对话
            "rag_docs",        # RAG 检索到的文档
            "tool_results",    # 工具返回结果
            "history_summary", # 历史摘要（最低优先级）
        ]
        self.min_budget = {
            "system": 200,
            "recent_history": 300,
            "rag_docs": 200,
            "tool_results": 100,
            "history_summary": 50,
        }

    def fit_into_budget(
            self,
            components: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                component_tokens = self._count_component_tokens(components)
    
                total_tokens = sum(component_tokens.values())
                if total_tokens <= self.total_budget:
                    return components, self._generate_report(components, component_tokens, "✅ 未超出预算")
                print(f"💰 [BudgetManager] 总 Token {total_tokens} 超出预算 {self.total_budget}，开始裁剪...")
    
                trimmed = self._copy_components(components)
                trimmed_tokens = component_tokens.copy()
    
                for priority in reversed(self.priority_order):
                    if priority not in trimmed or trimmed[priority] is None:
                        continue
                    current_total = sum(trimmed_tokens.values())
                    if current_total <= self.total_budget:
                        break
                    target = self.total_budget
                    excess = current_total - target
    
                    if priority == "history_summary":
                        trimmed[priority], trimmed_tokens[priority] = self._trim_summary(
                        trimmed[priority], excess, priority
                    )
                    elif priority == "tool_results":
                        trimmed[priority], trimmed_tokens[priority] = self._trim_list(
                            trimmed[priority], excess, priority
                        )
                    elif priority == "rag_docs":
                        trimmed[priority], trimmed_tokens[priority] = self._trim_list(
                        trimmed[priority], excess, priority
                    )
                    elif priority == "recent_history":
                        trimmed[priority], trimmed_tokens[priority] = self._trim_messages(
                        trimmed[priority], excess, priority
                    )
                    elif priority == "system":
                        trimmed[priority], trimmed_tokens[priority] = self._trim_text(
                        trimmed[priority], excess, priority
                    )
                    print(f"  💰 [BudgetManager] 裁剪 {priority}: {component_tokens[priority]} -> {trimmed_tokens[priority]} tokens")
                final_total = sum(trimmed_tokens.values())
                status = "✅ 已裁剪到预算内" if final_total <= self.total_budget else "⚠️ 仍超出预算（各组件已达最小保留值）"
                return trimmed, self._generate_report(trimmed, trimmed_tokens, status)

    def _count_component_tokens(self, components: Dict[str, Any]) -> Dict[str, int]:
        result = {}
        for key, value in components.items():
            if value is None:
                result[key] = 0
                continue

            if key == "system":
                result[key] = count_tokens(value, self.model)
            elif key == "history_summary":
                result[key] = count_tokens(value, self.model)
            elif key in ["recent_history", "tool_results"]:
                if isinstance(value, list):
                    result[key] = count_messages_tokens(value, self.model)
                else:
                    result[key] = 0
            elif key == "rag_docs":
                if isinstance(value, list):
                    total = 0
                    for doc in value:
                        content = doc.get("content", "")
                        total += count_tokens(content, self.model)
                    result[key] = total
                else:
                    result[key] = 0
            else:
                result[key] = 0
        return result

    def _trim_text(self, text: str, excess: int, priority: str) -> Tuple[str, int]:
        if not text:
            return text, 0

        min_tokens = self.min_budget.get(priority, 50)
        current_tokens = count_tokens(text, self.model)

        if current_tokens <= min_tokens:
            return text, current_tokens
        target_tokens = max(min_tokens, current_tokens - excess)

        enc = self.encoding
        tokens = enc.encode(text)
        if len(tokens) <= target_tokens:
            return text, current_tokens

        trimmed_tokens = tokens[:target_tokens]
        trimmed_text = enc.decode(trimmed_tokens) + "... (已截断)"
        return trimmed_text, len(trimmed_tokens)

    def _trim_messages(self, messages: List[Dict], excess: int, priority: str) -> Tuple[List[Dict], int]:
        if not messages:
            return messages, 0

        min_tokens = self.min_budget.get(priority, 100)
        current_tokens = count_messages_tokens(messages, self.model)

        if current_tokens <= min_tokens:
            return messages, current_tokens

        if len(messages) <= 2:
            return messages, current_tokens

        remove_count = max(1, len(messages) // 3)
        trimmed = messages[remove_count:]
        return trimmed, count_messages_tokens(trimmed, self.model)

    def _trim_list(self, items: List[Dict], excess: int, priority: str) -> Tuple[List[Dict], int]:
        if not items:
            return items, 0
        min_tokens = self.min_budget.get(priority, 100)
        current_tokens = 0
        for item in items:
            content = item.get("content", "")
            current_tokens += count_tokens(content, self.model)
        if current_tokens <= min_tokens:
            return items, current_tokens
        if len(items) <= 2:
            return items, current_tokens
        remove_count = max(1, len(items) // 3)
        trimmed = items[:-remove_count] if len(items) > remove_count else items[:1]

        trimmed_tokens = 0
        for item in trimmed:
            content = item.get("content", "")
            trimmed_tokens += count_tokens(content, self.model)
        return trimmed, trimmed_tokens

    def _trim_summary(self, summary: str, excess: int, priority: str) -> Tuple[str, int]:
        return self._trim_text(summary, excess, priority)

    def _copy_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(components)
    def _generate_report(
        self,
        components: Dict[str, Any],
        token_counts: Dict[str, int],
        status: str
    ) -> Dict[str, Any]:
        total = sum(token_counts.values())

        return {
            "status": status,
            "total_tokens": total,
            "budget_limit": self.total_budget,
            "usage_percentage": round(total / self.total_budget * 100, 1),
            "component_details": token_counts,
            "over_budget": total > self.total_budget,
        }

_global_budget_manager:Optional[TokenBudgetManager] = None

def get_budget_manager(
    total_budget: int = 4000,
    model: str = "gpt-4",
) -> TokenBudgetManager:
    """获取全局预算管理器"""
    global _global_budget_manager
    if _global_budget_manager is None:
        _global_budget_manager = TokenBudgetManager(
            total_budget=total_budget,
            model=model,
        )
    return _global_budget_manager