# test_filter.py

from src.context.filter import get_context_filter

def test_filter():
    filter = get_context_filter(min_doc_score=0.5, max_docs=3, max_conversation_turns=2)

    # 模拟消息
    messages = [
        {"role": "system", "content": "你是一个科研助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"},
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的模型..."},
        {"role": "user", "content": "它和 RNN 有什么区别？"},
        {"role": "assistant", "content": "RNN 是递归的，Transformer 是并行的..."},
    ]

    # 模拟文档
    docs = [
        {"content": "文档1", "score": 0.9},
        {"content": "文档2", "score": 0.6},
        {"content": "文档3", "score": 0.4},
        {"content": "文档4", "score": 0.8},
    ]

    result = filter.filter_context(messages, docs)

    print("=" * 50)
    print("📊 过滤统计")
    print("=" * 50)
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("📝 过滤后的消息")
    print("=" * 50)
    for msg in result["messages"]:
        print(f"  [{msg['role']}]: {msg['content'][:30]}...")

    print("\n" + "=" * 50)
    print("📄 过滤后的文档")
    print("=" * 50)
    for doc in result["docs"]:
        print(f"  - {doc['content']} (score: {doc['score']})")

if __name__ == "__main__":
    test_filter()