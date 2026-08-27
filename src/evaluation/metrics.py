from typing import Dict, Any, List


class MetricsCollector:
    """
    性能指标收集器
    指标：Latency、Token、Tool Calls
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_latency_ms = 0
        self.llm_calls = []
        self.tool_calls = []
        self.rag_calls = []
        self.memory_calls = []

    def collect_from_trace(self, trace_report: Dict):
        """从追踪报告中提取指标"""
        self._walk(trace_report)

    def _walk(self, node: Dict, depth: int = 0):
        """递归遍历追踪树"""
        node_type = node.get("type", "")
        
        if node_type == "llm":
            self.llm_calls.append({
                "name": node.get("name"),
                "duration_ms": node.get("duration_ms", 0),
                "tokens": node.get("metadata", {}).get("tokens", 0),
            })
        elif node_type == "tool":
            self.tool_calls.append({
                "name": node.get("name"),
                "duration_ms": node.get("duration_ms", 0),
            })
        elif node_type == "rag":
            self.rag_calls.append({
                "name": node.get("name"),
                "duration_ms": node.get("duration_ms", 0),
                "docs_found": node.get("metadata", {}).get("docs_found", 0),
            })
        elif node_type == "memory":
            self.memory_calls.append({
                "name": node.get("name"),
                "duration_ms": node.get("duration_ms", 0),
            })
        elif node_type == "root":
            self.total_latency_ms = node.get("duration_ms", 0)

        for child in node.get("children", []):
            self._walk(child, depth + 1)

    def get_summary(self) -> Dict:
        """获取指标汇总"""
        total_llm = len(self.llm_calls)
        total_tools = len(self.tool_calls)
        total_rag = len(self.rag_calls)
        total_memory = len(self.memory_calls)
        
        avg_llm_duration = sum(c["duration_ms"] for c in self.llm_calls) / total_llm if total_llm > 0 else 0
        avg_tool_duration = sum(c["duration_ms"] for c in self.tool_calls) / total_tools if total_tools > 0 else 0
        total_tokens = sum(c.get("tokens", 0) for c in self.llm_calls)

        return {
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_tokens": total_tokens,
            "llm_calls": total_llm,
            "tool_calls": total_tools,
            "rag_calls": total_rag,
            "memory_calls": total_memory,
            "avg_llm_duration_ms": round(avg_llm_duration, 2),
            "avg_tool_duration_ms": round(avg_tool_duration, 2),
        }

    def print_summary(self):
        """打印指标汇总"""
        s = self.get_summary()
        print("\n" + "=" * 50)
        print("📊 性能指标")
        print("=" * 50)
        print(f"  总耗时: {s['total_latency_ms']:.0f}ms")
        print(f"  LLM 调用: {s['llm_calls']} 次 (总 Token: {s['total_tokens']})")
        print(f"  工具调用: {s['tool_calls']} 次")
        print(f"  RAG 检索: {s['rag_calls']} 次")
        print(f"  记忆查询: {s['memory_calls']} 次")
        print(f"  平均 LLM 耗时: {s['avg_llm_duration_ms']:.0f}ms")