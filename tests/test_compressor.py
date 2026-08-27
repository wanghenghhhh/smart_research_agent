# test_compressor.py

import asyncio
from src.context.compressor import get_context_compressor

async def test_compressor():
    compressor = get_context_compressor(max_context_tokens=500)  # 设置小阈值，强制压缩

    # 模拟一个长对话
    messages = [
        {"role": "system", "content": "你是一个科研助手"},
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的模型..." * 20},
        {"role": "user", "content": "它和 RNN 有什么区别？"},
        {"role": "assistant", "content": "RNN 是递归的，Transformer 是并行的..." * 20},
        {"role": "user", "content": "那它有什么应用场景？"},
        {"role": "assistant", "content": "NLP、CV、多模态等领域..." * 20},
    ]

    result = await compressor.compress_context(messages)

    print("=" * 50)
    print("📊 压缩统计")
    print("=" * 50)
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("📝 压缩后的消息（查看摘要是否生成）")
    print("=" * 50)
    for msg in result["messages"]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:150]
        print(f"  [{role}]: {content}...")

if __name__ == "__main__":
    asyncio.run(test_compressor())