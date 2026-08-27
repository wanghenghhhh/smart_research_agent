from typing import Dict, List, Optional
import uuid
import os  # ✅ 补充：确保目录存在

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config.settings import settings
from src.retrieval.embeddings import EmbeddingModel

# 存储全局单例
_global_qdrant_store_instance = None


class QdrantStore:
    """
    Qdrant 向量数据库封装（单例模式）。
    """

    def __new__(cls, *args, **kwargs):
        global _global_qdrant_store_instance
        if _global_qdrant_store_instance is None:
            _global_qdrant_store_instance = super().__new__(cls)
        return _global_qdrant_store_instance

    def __init__(
        self,
        use_memory: bool = False,
        storage_path: Optional[str] = None,
    ):
        # 内部防二次初始化标记
        if hasattr(self, "_initialized") and self._initialized:
            print("♻️ 复用已有 QdrantStore 实例")
            return


        if use_memory:
            self.client = QdrantClient(":memory:")
            print("🧠 Qdrant 使用内存模式")
        else:
            if storage_path is None:
                storage_path = "./qdrant_storage"

            # ✅ 确保目录存在
            os.makedirs(storage_path, exist_ok=True)

            self.client = QdrantClient(path=storage_path)
            print(f"💾 Qdrant 使用本地存储: {storage_path}")


        self.collection_name = "research_knowledge"
        self.embedder = EmbeddingModel()
        self.vector_size = self.embedder.vector_size


        self._init_collection()

        # 标记已完成初始化，防止重复运行
        self._initialized = True
        print("✅ QdrantStore 初始化完成（单例）")


    def _init_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            print(f"📁 Collection 不存在，开始创建: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Collection 创建成功: {self.collection_name}")
        else:
            print(f"📁 Collection 已存在: {self.collection_name}")


    def upsert_documents(self, documents: List[Dict]):
        if not documents:
            print("⚠️ 没有需要写入的文档")
            return

        texts = [doc["content"] for doc in documents]
        print(f"🔄 正在生成 Embedding，文档数量: {len(texts)}")
        vectors = self.embedder.encode(texts)

        points = []
        for i, doc in enumerate(documents):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "content": doc["content"],
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {}),
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        print(f"✅ 成功插入 {len(points)} 条知识")

    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        if not query or not query.strip():
            return []

        query_vector = self.embedder.encode(query)

        if query_vector.ndim == 2:
            query_vector = query_vector[0]

        try:
            # ✅ 优先使用 search（兼容旧版）
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=k,
                score_threshold=score_threshold,
                with_payload=True,  # ✅ 确保返回 payload
            )
        except AttributeError:
            # ✅ 如果 search 不可用，使用 query_points（新版 API）
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=k,
                score_threshold=score_threshold,
                with_payload=True,
            ).points

        output = []
        for hit in results:
            payload = hit.payload or {}
            output.append(
                {
                    "content": payload.get("content", ""),
                    "source": payload.get("source", "unknown"),
                    "score": hit.score,
                    "metadata": payload.get("metadata", {}),
                }
            )

        return output

    def count(self) -> int:
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
            return result.count
        except Exception as e:
            print(f"⚠️ 获取知识库数量失败: {e}")
            return 0


    def clear(self):
        print(f"⚠️ 正在清空知识库: {self.collection_name}")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            print(f"⚠️ 删除 Collection 时出现问题: {e}")

        self._init_collection()
        print("✅ 知识库已清空并重新创建")