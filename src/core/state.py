from typing import List, Dict, Optional, Any, TypedDict, Annotated
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    """
    MVP Agent的状态定义。
    TypedDict用于类型提示，确保状态字段符合预期。
    Annotated[..., add_messages] 是LangGraph的特殊标记，用于合并消息列表。
    """
    # 用户输入的研究问题
    query: str
    
    # 消息历史（LangGraph内置支持，自动合并新消息）
    messages: Annotated[List[Dict], add_messages]
    session_id:str
    intent_type:str
    
    # 研究计划文本（由Planner生成）
    research_plan: Optional[str]
    
    # 任务列表，每个任务是一个字典，包含id, type, content, status
    tasks: List[Dict[str, Any]]
    
    # 当前正在执行的任务索引
    current_task_index: int
    
    # 已完成的任务记录（历史）
    task_history: List[Dict]
    
    # 所有搜索结果的汇总
    search_results: List[Dict]
    
    retrieved_docs: List[Dict]#检索相关
    search_results: List[Dict]
    query_complexity: Optional[float]
    adaptive_k: Optional[int]
    retrieval_source: str

    # 每一步工具调用的记录
    intermediate_steps: List[Dict]

    is_memory_query: bool  # 是否是记忆查询
    memory_answer: Optional[str]
    #状态控制
    research_complete: bool
    final_report: Optional[str]
    error: Optional[str]