from typing import Dict, List, Any
from src.evaluation.metrics import MetricsCollector


class SimpleEvaluator:
    """
    简单评估器
    评估维度：Answer Quality、Retrieval Quality、Tool Success Rate
    """
    
    def __init__(self):
        self.results = []

    def evaluate(
        self,
        query: str,
        final_answer: str,
        trace_report: Dict,
        expected_keywords: List[str] = None,
        expected_tool: str = None,
        is_rag_query: bool = False,
    ) -> Dict:
        """
        评估单个查询
        
        参数：
            query: 用户问题
            final_answer: Agent 最终回答
            trace_report: 追踪报告
            expected_keywords: 期望出现的关键词
            expected_tool: 期望调用的工具
            is_rag_query: 是否期望触发 RAG
        """
        # 1. Answer Quality：关键词匹配
        answer_quality = self._check_keywords(final_answer, expected_keywords or [])
        
        # 2. Retrieval Quality：检查是否有 RAG 检索
        retrieval_quality = self._check_rag(trace_report, is_rag_query)
        
        # 3. Tool Success Rate：检查工具是否被调用
        tool_success = self._check_tool(trace_report, expected_tool)
        
        result = {
            "query": query,
            "answer_quality": answer_quality,
            "retrieval_quality": retrieval_quality,
            "tool_success": tool_success,
            "overall": answer_quality and retrieval_quality and tool_success,
        }
        self.results.append(result)
        return result

    def _check_keywords(self, text: str, keywords: List[str]) -> bool:
        """检查是否包含所有关键词"""
        if not keywords:
            return True
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() not in text_lower:
                return False
        return True

    def _check_rag(self, trace_report: Dict, is_rag_query: bool) -> bool:
        """检查是否有 RAG 检索（如果是 RAG 场景）"""
        if not is_rag_query:
            return True
        
        def walk(node):
            if node.get("type") == "rag":
                return True
            for child in node.get("children", []):
                if walk(child):
                    return True
            return False
        
        return walk(trace_report)

    def _check_tool(self, trace_report: Dict, expected_tool: str) -> bool:
        """检查是否调用了期望的工具"""
        if not expected_tool:
            return True
        
        def walk(node):
            if node.get("type") == "tool" and node.get("name") == expected_tool:
                return True
            if node.get("type") == "tool" and expected_tool in node.get("name", ""):
                return True
            for child in node.get("children", []):
                if walk(child):
                    return True
            return False
        
        return walk(trace_report)

    def get_summary(self) -> Dict:
        """获取评估汇总"""
        total = len(self.results)
        if total == 0:
            return {"total": 0, "overall_rate": 0}
        
        answer_passed = sum(1 for r in self.results if r["answer_quality"])
        retrieval_passed = sum(1 for r in self.results if r["retrieval_quality"])
        tool_passed = sum(1 for r in self.results if r["tool_success"])
        overall_passed = sum(1 for r in self.results if r["overall"])

        return {
            "total": total,
            "answer_quality_rate": round(answer_passed / total * 100, 1),
            "retrieval_quality_rate": round(retrieval_passed / total * 100, 1),
            "tool_success_rate": round(tool_passed / total * 100, 1),
            "overall_rate": round(overall_passed / total * 100, 1),
        }

    def print_report(self):
        """打印评估报告"""
        s = self.get_summary()
        print("\n" + "=" * 50)
        print("📊 评估报告")
        print("=" * 50)
        print(f"  测试总数: {s['total']}")
        print(f"  Answer Quality: {s['answer_quality_rate']:.1f}%")
        print(f"  Retrieval Quality: {s['retrieval_quality_rate']:.1f}%")
        print(f"  Tool Success Rate: {s['tool_success_rate']:.1f}%")
        print(f"  Overall: {s['overall_rate']:.1f}%")