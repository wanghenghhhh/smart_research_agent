# src/utils/retry.py

import asyncio
import time
from functools import wraps
from typing import Type, Tuple, Optional


def with_retry_and_timeout(
    max_retries: int = 2,
    base_delay: float = 1.0,
    timeout: float = 20.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    装饰器：为异步函数增加重试和超时机制
    
    参数：
        max_retries: 最大重试次数（不包括首次执行）
        base_delay: 基础延迟时间（秒），每次重试翻倍（指数退避）
        timeout: 单次执行的超时时间（秒）
        exceptions: 需要捕获并重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # 1. 执行函数，并设置超时
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    # 如果成功，打印重试成功信息（如果有重试）
                    if attempt > 0:
                        print(f"✅ 工具 {func.__name__} 重试成功 (尝试 {attempt+1}/{max_retries+1})")
                    
                    return result
                
                except asyncio.TimeoutError as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⏰ 工具 {func.__name__} 超时 ({timeout}s)，重试 {attempt+1}/{max_retries}...")
                        await asyncio.sleep(base_delay * (attempt + 1))
                    else:
                        raise TimeoutError(f"工具 {func.__name__} 在重试 {max_retries} 次后仍然超时") from e
                
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"🔄 工具 {func.__name__} 失败: {str(e)[:50]}，重试 {attempt+1}/{max_retries}...")
                        await asyncio.sleep(base_delay * (attempt + 1))
                    else:
                        # 重试耗尽，抛出最终异常
                        raise Exception(f"工具 {func.__name__} 在重试 {max_retries} 次后仍然失败: {str(e)}") from e
            
            # 理论上不会执行到这里，但为了安全
            raise last_exception
        
        return wrapper
    
    return decorator