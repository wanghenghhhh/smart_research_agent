from src.retrieval.qdrant_store import QdrantStore


def main():
    print("=" * 60)
    print("开始测试 Qdrant 文档入库")
    print("=" * 60)

    # 初始化 Qdrant
    store = QdrantStore(
        storage_path="./qdrant_storage"
    )

    # 准备测试知识
    documents = [
        {
            "content": (
                "Transformer 是一种基于注意力机制的深度学习模型架构，"
                "最初由 Vaswani 等人在 2017 年提出。"
                "Transformer 使用 Self-Attention 机制处理序列数据。"
            ),
            "source": "test_transformer",
            "metadata": {
                "topic": "Transformer",
                "type": "test"
            }
        },
        {
            "content": (
                "Self-Attention 是 Transformer 的核心机制之一。"
                "它能够计算序列中不同 token 之间的关联程度，"
                "从而建立不同位置之间的信息依赖关系。"
            ),
            "source": "test_attention",
            "metadata": {
                "topic": "Attention",
                "type": "test"
            }
        },
        {
            "content": (
                "RAG，即 Retrieval-Augmented Generation，"
                "是一种将信息检索与大语言模型生成结合起来的方法。"
                "RAG 通常包括文档切分、向量化、向量检索和生成等步骤。"
            ),
            "source": "test_rag",
            "metadata": {
                "topic": "RAG",
                "type": "test"
            }
        },
        {
            "content": (
                "向量数据库可以保存文本对应的向量表示，"
                "并通过向量相似度计算找到与查询最相关的文档。"
                "Qdrant 是一种支持向量相似度检索的向量数据库。"
            ),
            "source": "test_vector_database",
            "metadata": {
                "topic": "Vector Database",
                "type": "test"
            }
        },
    ]

    # ============================================================
    # 写入 Qdrant
    # ============================================================

    store.upsert_documents(documents)

    # ============================================================
    # 查看当前知识库数量
    # ============================================================

    count = store.count()

    print()
    print(f"📚 当前知识库文档数量：{count}")

    print("=" * 60)
    print("Qdrant 文档入库测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()