import asyncio

from src.tools.definitions import search_knowledge_base


async def main():

    print("=" * 60)
    print("开始测试 Adaptive RAG Tool")
    print("=" * 60)

    query = "什么是 Transformer？"

    print(f"\n用户问题：{query}")

    result = await search_knowledge_base.ainvoke(
        {
            "query": query
        }
    )

    print("\n" + "=" * 60)
    print("Tool 返回结果")
    print("=" * 60)

    print(result)

    print("\n" + "=" * 60)
    print("Adaptive RAG Tool 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())