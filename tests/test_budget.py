# test_budget.py

from src.context.budget import get_budget_manager


def test_budget():
    manager = get_budget_manager(total_budget=2000)  # 设小一点，强制触发裁剪

    # 构造模拟组件
    components = {
        "system": "你是一个科研助手。请根据以下信息回答问题。",  # ~20 tokens
        "recent_history": [
            {"role": "user", "content": "什么是 Transformer？"},
            {"role": "assistant", "content": "Transformer 是一种..." * 50},  # 制造长文本
            {"role": "user", "content": "它有什么优势？"},
        ],
        "rag_docs": [
            {"content": "文档1: " + "A" * 500},
            {"content": "文档2: " + "B" * 500},
            {"content": "文档3: " + "C" * 500},
        ],
        "tool_results": [
            {"content": "工具返回: " + "X" * 300},
            {"content": "工具返回: " + "Y" * 300},
        ],
        "history_summary": "用户问了 Transformer 的基本概念和优势。" * 10,
    }

    print("=" * 50)
    print("💰 测试 Token 预算管理")
    print("=" * 50)

    trimmed, report = manager.fit_into_budget(components)

    print(f"\n📊 预算报告:")
    print(f"  状态: {report['status']}")
    print(f"  总 Token: {report['total_tokens']} / {report['budget_limit']}")
    print(f"  使用率: {report['usage_percentage']}%")
    print(f"\n  各组件详情:")
    for key, value in report['component_details'].items():
        print(f"    {key}: {value} tokens")

    print(f"\n📋 裁剪后组件数量:")
    print(f"  system: {len(trimmed.get('system', ''))} 字符")
    print(f"  recent_history: {len(trimmed.get('recent_history', []))} 条消息")
    print(f"  rag_docs: {len(trimmed.get('rag_docs', []))} 篇")
    print(f"  tool_results: {len(trimmed.get('tool_results', []))} 条")
    print(f"  history_summary: {len(trimmed.get('history_summary', ''))} 字符")


if __name__ == "__main__":
    test_budget()