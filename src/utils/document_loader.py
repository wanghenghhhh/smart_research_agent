# src/utils/document_loader.py
from src.retrieval.qdrant_store import QdrantStore

def load_sample_knowledge():
    # 使用本地存储，强制重建
    store = QdrantStore(storage_path="./qdrant_storage", force_recreate=True)
    
    docs = [
        {"content": "Transformer 是一种基于自注意力机制的深度学习模型，由 Vaswani 等人在 2017 年提出。它解决了 RNN 无法并行计算的瓶颈。", "source": "knowledge_base"},
        {"content": "RNN（循环神经网络）擅长处理序列数据，如文本和时间序列。但存在梯度消失问题，LSTM 和 GRU 是其改进版本。", "source": "knowledge_base"},
        {"content": "RAG（检索增强生成）结合了信息检索和生成式大模型，能有效减少幻觉，提供可溯源的回答。", "source": "knowledge_base"},
        {"content": "Qdrant 是一个高性能的向量数据库，使用 Rust 编写，支持余弦相似度和点积，常用于 AI 应用的向量检索。", "source": "knowledge_base"},
        {"content": "LangGraph 是 LangChain 生态中用于构建有状态多角色 Agent 的框架，基于图结构实现复杂流程编排。", "source": "knowledge_base"},
        {"content": "Adaptive RAG 是一种根据查询复杂度动态调整检索策略的方法，简单问题少检索节省成本，复杂问题多检索并分解查询。", "source": "knowledge_base"},
    ]
    
    store.upsert_documents(docs)
    print("🎉 知识库种子数据加载完成！")

if __name__ == "__main__":
    load_sample_knowledge()