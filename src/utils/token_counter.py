# src/utils/token_counter.py

import tiktoken
from typing import List, Dict, Any

# 模型对应的编码器名称
MODEL_ENCODING_MAP = {
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "qwen-max": "cl100k_base",  # Qwen 使用同样的编码器估算
    "qwen-plus": "cl100k_base",
}


def get_encoding(model: str = "gpt-4"):
    """获取指定模型的编码器"""
    encoding_name = MODEL_ENCODING_MAP.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """计算单个文本的 token 数"""
    if not text:
        return 0
    enc = get_encoding(model)
    return len(enc.encode(text))


def count_messages_tokens(messages: List[Dict], model: str = "gpt-4") -> int:
    """计算消息列表的 token 数（模拟 OpenAI API 的计费方式）"""
    if not messages:
        return 0
    
    enc = get_encoding(model)
    total = 0
    
    for msg in messages:
        # 每条消息的固定开销（OpenAI API 计费规则）
        total += 4  # 每条消息的元数据开销
        
        # 消息内容
        content = msg.get("content", "")
        if content:
            total += len(enc.encode(content))
        
        # 角色
        role = msg.get("role", "")
        if role:
            total += len(enc.encode(role))
    
    # 回复的固定开销（OpenAI API 规则）
    total += 2
    
    return total


def should_compress(messages: List[Dict], threshold: int = 4000, model: str = "gpt-4") -> bool:
    """判断是否需要压缩上下文"""
    token_count = count_messages_tokens(messages, model)
    return token_count > threshold


def get_token_budget(messages: List[Dict], model: str = "gpt-4") -> dict:
    """获取 token 使用情况报告"""
    total = count_messages_tokens(messages, model)
    return {
        "total_tokens": total,
        "message_count": len(messages),
        "needs_compression": total > 4000,
    }