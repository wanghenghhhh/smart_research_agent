# src/observability/tracer.py

import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import contextmanager


class TraceSpan:
    """追踪的一个跨度（一次操作）"""
    
    def __init__(self, name: str, span_type: str):
        self.name = name
        self.span_type = span_type  # planner, tool, rag, memory, reporter, llm
        self.start_time = time.time()
        self.end_time = None
        self.duration_ms = None
        self.metadata = {}
        self.children = []

    def finish(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.span_type,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else 0,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class AgentTracer:
    """
    Agent 追踪器
    记录：Planner / Tool / RAG / Memory / Reporter 的调用链
    """
    
    def __init__(self):
        self.root = None
        self.current = None
        self.stack = []
        self.events = []

    def start_trace(self, query: str):
        """开始一次追踪"""
        self.root = TraceSpan(f"Research: {query[:30]}...", "root")
        self.current = self.root
        self.stack = [self.root]
        self.events = []
        print(f"\n🔍 [Trace] 开始追踪: {query[:50]}...")

    def start_span(self, name: str, span_type: str, metadata: Dict = None):
        """开始一个子操作"""
        span = TraceSpan(name, span_type)
        if metadata:
            span.metadata = metadata
        if self.current:
            self.current.children.append(span)
        self.stack.append(span)
        self.current = span
        
        indent = "  " * (len(self.stack) - 1)
        icons = {"planner": "📋", "tool": "🛠️", "rag": "📚", "memory": "🧠", "reporter": "📝", "llm": "🤖"}
        icon = icons.get(span_type, "🔹")
        print(f"{indent}└─ {icon} {name}")

    def end_span(self):
        """结束当前操作"""
        if self.current:
            self.current.finish()
            self.stack.pop()
            self.current = self.stack[-1] if self.stack else None

    def add_event(self, name: str, data: Any = None):
        """记录一个事件（不创建新层级）"""
        self.events.append({"name": name, "data": data, "time": time.time()})
        indent = "  " * len(self.stack)
        if data:
            print(f"{indent}   📌 {name}: {str(data)[:100]}")
        else:
            print(f"{indent}   📌 {name}")

    def end_trace(self):
        """结束追踪，打印摘要"""
        if self.root:
            self.root.finish()
            print(f"\n📊 [Trace] 总耗时: {self.root.duration_ms:.0f}ms")
            self._print_tree(self.root)

    def _print_tree(self, span: TraceSpan, indent: str = ""):
        for child in span.children:
            dur = f"{child.duration_ms:.0f}ms"
            print(f"{indent}  ├─ {child.name} ({dur})")
            if child.children:
                self._print_tree(child, indent + "  ")

    def get_report(self) -> Dict:
        """获取追踪报告（JSON 格式）"""
        if self.root:
            return self.root.to_dict()
        return {"name": "empty", "children": []}


# 全局单例
_tracer = None

def get_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer