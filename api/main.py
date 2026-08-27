# api/main.py

import json
import uuid
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from api.models import ResearchRequest, ResearchResponse
from src.core.workflow import ResearchWorkflow
from src.evaluation.tracer import get_tracer
from src.evaluation.metrics import MetricsCollector

_workflow = None

def get_workflow():
    global _workflow
    if _workflow is None:
        print("🚀 正在初始化 Agent 工作流...")
        _workflow = ResearchWorkflow(db_path="checkpoints.db")
    return _workflow

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔧 启动 FastAPI 服务...")
    get_workflow()
    print("✅ Agent 已就绪，等待请求...")
    yield
    if _workflow:
        _workflow.close()
        print("👋 资源已清理")

app = FastAPI(
    title="Smart Research Agent API",
    description="基于 LangGraph + ReAct + RAG 的智能科研助手",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/chat", response_model=ResearchResponse)
async def chat(request: ResearchRequest):
    workflow = get_workflow()
    thread_id = request.thread_id
    if not thread_id:
        thread_id = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    try:
        result = await workflow.run(request.query, thread_id=thread_id)
        answer = result.get("final_report", "Agent 未能生成有效回答。")
        error = result.get("error")

        metrics = None
        trace_id = None
        if request.trace:
            tracer = get_tracer()
            trace_report = tracer.get_report()
            trace_id = f"trace_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            metrics_collector = MetricsCollector()
            metrics_collector.collect_from_trace(trace_report)
            metrics = metrics_collector.get_summary()
            metrics["trace_tree"] = trace_report

        return ResearchResponse(
            success=error is None,
            answer=answer,
            thread_id=thread_id,
            trace_id=trace_id,
            metrics=metrics,
            error=error,
        )
    except Exception as e:
        return ResearchResponse(
            success=False,
            answer="",
            thread_id=thread_id,
            error=str(e),
        )

@app.post("/chat/stream")
async def chat_stream(request: ResearchRequest):
    workflow = get_workflow()

    thread_id = request.thread_id
    if not thread_id:
        thread_id = f"api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # 1. 发送会话 ID
            yield f"data: {json.dumps({'type': 'session', 'thread_id': thread_id})}\n\n"

            # 2. 流式执行 Agent
            async for chunk in workflow.run_stream(request.query, thread_id=thread_id):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )