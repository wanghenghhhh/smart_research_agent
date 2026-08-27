from src.retrieval.qdrant_store import QdrantStore


def main():
    print("=" * 60)
    print("开始测试 Qdrant 向量检索")
    print("=" * 60)

    store = QdrantStore(
        storage_path="./qdrant_storage"
    )

    query = "什么是 Transformer？"

    print(f"\n🔎 查询：{query}")

    results = store.search(
        query=query,
        k=3,
        score_threshold=0.0
    )

    print(f"\n📚 检索到 {len(results)} 条结果\n")

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"相似度：{result['score']:.4f}")
        print(f"来源：{result.get('source', 'unknown')}")
        print(f"内容：{result['content']}")
        print()

    print("=" * 60)
    print("Qdrant 向量检索测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()