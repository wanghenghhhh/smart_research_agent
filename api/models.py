from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ResearchRequest(BaseModel):
    query: str = Field(..., description="用户的研究问题", example="大模型应用开发需要学习哪些技术？")
    thread_id: Optional[str] = Field(
        default=None,
        description="会话 ID（留空自动生成，用于记忆延续）",
        example="interview_demo"
    )
    trace: bool = Field(
        default=True,
        description="是否开启详细追踪日志"
    )

class ResearchResponse(BaseModel):
    success: bool
    answer: str
    thread_id: str
    trace_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None