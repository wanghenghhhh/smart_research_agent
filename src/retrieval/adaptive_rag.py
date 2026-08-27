from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from config.settings import settings
from src.retrieval.qdrant_store import QdrantStore

# 存储全局单例
_global_rag_instance = None


class AdaptiveRAG:
    """
    自适应 RAG 检索器（单例模式）。
    """

    def __new__(cls, *args, **kwargs):
        global _global_rag_instance
        if _global_rag_instance is None:
            _global_rag_instance = super().__new__(cls)
        return _global_rag_instance

    def __init__(self):
        # 内部防二次初始化标记
        if hasattr(self, "_initialized") and self._initialized:
            print("♻️ 复用已有 AdaptiveRAG 实例")
            return

        print("🔄 正在初始化 Adaptive RAG (单例)...")
        self.store = QdrantStore(storage_path="./qdrant_storage")

        # ✅ 微调：base_url 空值时传 None
        base_url = settings.OPENAI_BASE_URL if settings.OPENAI_BASE_URL else None

        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            base_url=base_url,
        )


        self.min_k = 3
        self.max_k = 15

        self._initialized = True
        print("✅ Adaptive RAG 初始化成功（已常驻内存）")

    async def retrieve(self, query: str) -> Tuple[List[Dict], int]:
        if not query or not query.strip():
            return [], self.min_k

        # 1. 分析问题复杂度
        complexity = await self._analyze_complexity(query)

        # 2. 根据复杂度计算 K
        k = int(self.min_k + (self.max_k - self.min_k) * complexity)
        k = max(self.min_k, min(self.max_k, k))

        # 3. 根据复杂度选择检索策略
        if complexity > 0.75:
            print(f"🧠 检测到高复杂度问题 (complexity={complexity:.2f})")
            docs = await self._complex_search(query, k)
        else:
            print(f"🔎 直接进行向量检索 (complexity={complexity:.2f})")
            docs = self.store.search(query, k=k)

        print(f"📊 [Adaptive RAG] 复杂度={complexity:.2f}, K={k}, 命中={len(docs)}篇")

        return docs, k


    async def _analyze_complexity(self, query: str) -> float:
        prompt = f"""
你是一个科研问题复杂度分析器。

请判断下面科研问题的复杂度。

只返回一个 0 到 1 之间的数字。

0 = 非常简单
例如：
什么是 Transformer？

0.3 = 比较简单
例如：
Transformer 的核心组成是什么？

0.5 = 中等复杂
例如：
Transformer 和 RNN 有什么区别？

0.8 = 高复杂度
例如：
比较 Transformer 和 RNN 在多模态任务中的优缺点。

1.0 = 极高复杂度
例如：
综合分析 Transformer、RNN、CNN 在多模态大模型中的发展，
比较其性能、计算成本和未来研究趋势。

科研问题：

{query}

复杂度：
"""
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            score = float(content)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ 复杂度分析失败：{e}")
            return 0.5

    async def _complex_search(self, query: str, k: int) -> List[Dict]:
        # 1. 查询拆解
        sub_queries = await self._decompose_query(query)
        print(f"🔍 拆解为子查询：{sub_queries}")

        if not sub_queries:
            print("⚠️ 子查询拆解失败，退化为普通检索")
            return self.store.search(query, k=k)

        # 2. 每个子查询分别检索
        all_docs = []
        per_query_k = max(2, k // len(sub_queries))

        for sub_query in sub_queries:
            print(f"🔎 子查询检索：{sub_query}")
            docs = self.store.search(sub_query, k=per_query_k)
            all_docs.extend(docs)

        # 3. 去重
        seen = set()
        unique_docs = []

        for doc in all_docs:
            content = doc.get("content", "")
            key = content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        # 4. 根据相似度排序
        unique_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 5. 返回 Top-K
        return unique_docs[:k]

    async def _decompose_query(self, query: str) -> List[str]:
        prompt = f"""
你是一个科研问题拆解器。

请把下面的科研问题拆解成 2-4 个
可以独立进行知识库检索的子问题。

要求：

1. 每个子问题必须具体。
2. 每个子问题应该能够独立进行向量检索。
3. 每行一个子问题。
4. 不要解释。
5. 不要输出其他内容。

原始问题：

{query}

子问题：
"""
        try:
            response = await self.llm.ainvoke(prompt)
            lines = response.content.split("\n")
            clean_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = line.lstrip("-•* ")

                if len(line) >= 2 and line[0].isdigit() and line[1] in [".", "、", ")"]:
                    line = line[2:].strip()

                if line:
                    clean_lines.append(line)

            clean_lines = clean_lines[:4]
            return clean_lines if clean_lines else [query]

        except Exception as e:
            print(f"⚠️ 查询拆解失败：{e}")
            return [query]