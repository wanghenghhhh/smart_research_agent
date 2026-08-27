# src/tools/definitions.py

import datetime
import logging
import math

from langchain.tools import tool
from src.utils.retry import with_retry_and_timeout 
from src.tools.search import SearchTool
from src.retrieval.adaptive_rag import AdaptiveRAG

logger = logging.getLogger(__name__)
search_instance = SearchTool()
rag_instance = AdaptiveRAG()

@tool
@with_retry_and_timeout(
    max_retries=2,
    base_delay=1.5,
    timeout=15.0,        # 15秒超时
    exceptions=(Exception,)
)
async def search_web(query: str) -> str:
    """
    在互联网上搜索最新、实时的信息。

    适用于：
    - 最新论文
    - 最新新闻
    - 当前事件
    - 外部网站
    - 最新科研进展
    """

    try:

        results = search_instance.search(
            query,
            max_results=3
        )

        if not results:
            return "未找到相关信息。"

        output = []

        for i, result in enumerate(
            results,
            1
        ):

            title = result.get(
                "title",
                "无标题"
            )

            content = result.get(
                "content",
                ""
            )

            url = result.get(
                "url",
                ""
            )

            output.append(
                f"[搜索结果 {i}]\n"
                f"标题：{title}\n"
                f"内容：{content[:500]}\n"
                f"来源：{url}"
            )

        return "\n\n".join(output)

    except Exception as e:

        logger.exception(
            "Web Search 执行失败"
        )

        return (
            f"Web 搜索失败：{str(e)}"
        )


# ============================================================
# Adaptive RAG
# ============================================================

_rag_instance = None


def get_rag_instance() -> AdaptiveRAG:
    """
    获取 Adaptive RAG 实例。

    使用单例模式，避免每次调用 Tool 都重新加载
    Embedding 模型和 Qdrant。
    """
    global _rag_instance

    if _rag_instance is None:
        print("🔄 正在初始化 Adaptive RAG...")
        _rag_instance = AdaptiveRAG()
        print("✅ Adaptive RAG 初始化成功")

    return _rag_instance


# ============================================================
# Adaptive RAG Tool
# ============================================================

@tool
@with_retry_and_timeout(
    max_retries=1,
    base_delay=1.0,
    timeout=10.0,        # 15秒超时
    exceptions=(Exception,)
)
async def search_knowledge_base(query: str) -> str:
    """
    使用 Adaptive RAG 从科研知识库中检索与问题最相关的文档。

    适用于：
    - 科研论文知识查询
    - 技术概念查询
    - 已经存入知识库的研究资料查询
    - 需要从本地科研知识库获取背景信息的问题

    Args:
        query: 用户需要查询的问题。

    Returns:
        格式化后的知识库检索结果。
    """

    if not query or not query.strip():
        return "错误：查询问题不能为空。"

    try:
        rag = get_rag_instance()

        print(f"\n🔎 Adaptive RAG Tool 查询：{query}")

        # 调用你的 Adaptive RAG
        docs, used_k = await rag.retrieve(query)

        if not docs:
            return (
                f"知识库没有找到与问题相关的内容。\n"
                f"查询：{query}\n"
                f"实际检索 K：{used_k}"
            )

        results = []

        results.append(
            f"知识库检索结果\n"
            f"查询：{query}\n"
            f"实际检索数量 K：{used_k}\n"
            f"共找到 {len(docs)} 条相关文档。\n"
        )

        for i, doc in enumerate(docs, 1):

            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            score = doc.get("score", 0.0)

            results.append(
                f"--- 文档 {i} ---\n"
                f"相关度：{score:.4f}\n"
                f"来源：{source}\n"
                f"内容：{content}\n"
            )

        return "\n".join(results)

    except Exception as e:
        print(f"❌ Adaptive RAG Tool 执行失败：{e}")

        return (
            "知识库检索失败。\n"
            f"错误信息：{str(e)}"
        )
    
@tool
def calculate(
    expression: str
) -> str:
    """
    执行数学计算。

    示例：

    2+3*4
    sqrt(16)
    100/5
    """

    try:

        allowed_names = {
            k: v
            for k, v in math.__dict__.items()
            if not k.startswith("__")
        }

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            allowed_names
        )

        return (
            f"计算结果：{result}"
        )

    except Exception as e:

        return (
            f"计算错误：{str(e)}"
        )

@tool
def get_current_time() -> str:
    """
    获取当前日期和时间。
    """

    now = datetime.datetime.now()

    return now.strftime(
        "当前时间：%Y年%m月%d日 %H:%M:%S"
    )

TOOLS = [
    search_web,
    search_knowledge_base,
    calculate,
    get_current_time,
]


# ============================================================
# Standalone Test
# ============================================================

async def test_rag_tool():

    print()
    print("=" * 60)
    print("开始测试 Adaptive RAG Tool")
    print("=" * 60)

    query = "什么是 Transformer？"

    print()
    print(
        f"查询问题：{query}"
    )

    result = await search_knowledge_base.ainvoke(
        query
    )

    print()
    print("=" * 60)
    print("RAG TOOL 测试结果")
    print("=" * 60)

    print(result)

    print("=" * 60)


if __name__ == "__main__":

    import asyncio

    asyncio.run(
        test_rag_tool()
    )