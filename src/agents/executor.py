from typing import Dict, List
from src.tools.search import SearchTool
from src.retrieval.adaptive_rag import AdaptiveRAG

class Executor:
    """
    任务执行器：负责执行当前任务，并根据任务类型调用相应工具。
    目前支持 search 类型，其他类型（read, analyze）作为占位。
    """
    
    def __init__(self):
        # 实例化搜索工具
        self.search_tool = SearchTool()
        self.rag = AdaptiveRAG()
    
    async def execute(self, state: Dict) -> Dict:
        """
        执行当前任务
        
        参数:
            state: 当前状态
            
        返回:
            更新后的状态
        """
        task_index = state.get("current_task_index", 0)
        tasks = state.get("tasks", [])
        
        # 如果没有任务或已完成所有任务，标记完成
        if task_index >= len(tasks):
            state["research_complete"] = True
            return state
        
        current_task = tasks[task_index]
        task_type = current_task.get("type", "search")
        task_content = current_task.get("content", "")
        
        # 更新任务状态为运行中
        current_task["status"] = "running"
        
        try:
            # 根据任务类型执行不同的逻辑
            if task_type == "search":
                result = await self._execute_search(task_content,state)
            elif task_type == "read":
                result = await self._execute_read(task_content, state)
            elif task_type == "analyze":
                result = await self._execute_analyze(task_content, state)
            else:
                result = {"error": f"未知任务类型: {task_type}"}
            
            # 记录任务执行历史
            state["task_history"].append({
                "task_id": current_task["id"],
                "task_type": task_type,
                "content": task_content,
                "result": result,
                "status": "completed"
            })
            
            # 如果是搜索任务且有结果，存入 search_results
            if task_type == "search" and "results" in result:
                state["search_results"].extend(result["results"])
            
            # 标记任务完成，索引后移
            current_task["status"] = "completed"
            state["current_task_index"] = task_index + 1
            
            # 检查是否所有任务都完成
            if state["current_task_index"] >= len(tasks):
                state["research_complete"] = True
            
        except Exception as e:
            current_task["status"] = "failed"
            state["error"] = f"任务执行失败: {str(e)}"
        
        return state
    
    async def _execute_search(self, query: str,state:Dict) -> Dict:
        """
        执行搜索任务
        
        参数:
            query: 搜索查询
            
        返回:
            包含搜索结果和元数据的字典
        """
        kb_docs, used_k = await self.rag.retrieve(query)
        state["retrieved_docs"] = kb_docs
        state["adaptive_k"] = used_k
        final_results = kb_docs.copy()

        if len(kb_docs) < 2:
            print("🌐 知识库结果不足，触发网络搜索...")
            web_results = self.search_tool.search(query)
            for r in web_results:
                final_results.append({
                    "content": r["content"],
                    "source": r["url"],
                    "score": r["score"]
                    })

        return {
            "type": "search",
            "query": query,
            "results": final_results,
            "result_count": len(final_results),
            "source":"hybrid"
        }
    
    async def _execute_read(self, content: str, state: Dict) -> Dict:
        """
        执行阅读任务（MVP阶段简单实现）
        从已有搜索结果中查找与 content 相关的文档。
        """
        search_results = state.get("search_results", [])
        # 简单匹配：检查标题或内容是否包含查询词
        relevant_docs = []
        for result in search_results:
            if content.lower() in result.get("title", "").lower() or \
               content.lower() in result.get("content", "").lower():
                relevant_docs.append(result)
        
        return {
            "type": "read",
            "query": content,
            "found_docs": relevant_docs,
            "count": len(relevant_docs)
        }
    
    async def _execute_analyze(self, content: str, state: Dict) -> Dict:
        """
        执行分析任务（MVP阶段简单统计）
        """
        search_results = state.get("search_results", [])
        analysis = {
            "type": "analyze",
            "query": content,
            "total_results": len(search_results),
            "unique_sources": len(set(r.get("url", "") for r in search_results)),
            "topics": self._extract_topics(search_results)
        }
        return analysis
    
    def _extract_topics(self, results: List[Dict]) -> List[str]:
        """
        从结果中提取主题（简单方法：取标题前几个词）
        """
        topics = []
        for result in results[:5]:
            title = result.get("title", "")
            if title:
                words = title.split()[:3]
                topics.append(" ".join(words))
        return topics[:5]