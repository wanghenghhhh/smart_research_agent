from typing import List, Dict, Optional
from tavily import TavilyClient
from config.settings import settings

class SearchTool:
    """
    搜索工具类，封装 Tavily API 调用。
    Tavily 是一个专为AI设计的搜索引擎，返回结构化的搜索结果。
    """
    
    def __init__(self):
        # 初始化Tavily客户端，传入API密钥
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        # 默认最大结果数从配置读取
        self.max_results = settings.MAX_SEARCH_RESULTS
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        执行搜索
        
        参数:
            query: 搜索查询字符串
            max_results: 可选，覆盖默认的最大结果数
            
        返回:
            搜索结果列表，每个结果包含 title, url, content, score
        """
        if max_results is None:
            max_results = self.max_results
        
        try:
            # 调用 Tavily 的 search 方法
            # search_depth="basic" 表示基础搜索，速度较快，适合MVP阶段
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,      # 不返回总结答案（节省成本）
                include_raw_content=False  # 不返回原始网页内容（节省token）
            )
            
            # 提取结果列表
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),  # 摘要内容
                    "score": result.get("score", 0)        # 相关性评分
                })
            
            return results
            
        except Exception as e:
            # 打印错误但不中断程序，返回空列表
            print(f"搜索失败: {e}")
            return []