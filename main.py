import asyncio
import sys
import uuid
from datetime import datetime
from src.core.workflow import ResearchWorkflow
from config.settings import settings
import os

def get_thread_id():
    """获取或创建会话 ID"""
    print("\n" + "=" * 60)
    print("🔐 会话管理")
    print("=" * 60)
    print("提示：输入相同的会话 ID 可以继续之前的对话")
    print("      留空则自动生成新会话")
    print("-" * 60)
    
    thread_id = input("请输入会话 ID（留空自动生成）: ").strip()
    if not thread_id:
        thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        print(f"✅ 已生成新会话: {thread_id}")
    else:
        print(f"✅ 使用已有会话: {thread_id}")
    
    return thread_id

async def main():
    print("=" * 60)
    print("🔬 智能科研助手 - Phase 5: Memory")
    print("=" * 60)
    print(f"📁 输出目录: {settings.OUTPUT_DIR}")
    print(f"🤖 模型: {settings.OPENAI_MODEL}")
    print("💾 记忆存储: checkpoints.db (SQLite)")
    print("=" * 60)
    print()

    thread_id = get_thread_id()

    print("\n🚀 初始化工作流...")
    workflow = ResearchWorkflow(db_path="checkpoints.db")
    sessions = workflow.list_sessions()

    if len(sessions) > 1:
        print(f"\n📜 历史会话: {len(sessions)} 个")
        for s in sessions:
            if s != thread_id:
                print(f"   - {s}")
    print("\n" + "=" * 60)
    print("💬 进入对话模式（输入 'exit' 退出）")
    print("=" * 60)
    print()

    while True:
        try:
            # 获取用户输入
            query = input("👤 你: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break
            
            if query.lower() in ["clear", "清除"]:
                # 清除当前会话记忆
                workflow.delete_session(thread_id)
                print("✅ 当前会话记忆已清除")
                continue
            
            if query.lower() in ["sessions", "会话列表"]:
                sessions = workflow.list_sessions()
                print(f"📜 所有会话: {sessions}")
                continue
            
            # 执行
            print("\n🤖 Agent 思考中...")
            print("-" * 60)
            
            result = await workflow.run(query, thread_id=thread_id)
            
            if result.get("error"):
                print(f"❌ 错误: {result['error']}")
                continue
            
            # 显示结果
            final_report = result.get("final_report", "")
            
            if final_report:
                print("\n📄 回答:\n")
                print(final_report)
            else:
                print("⚠️ 未生成有效回答")
            
            # 如果是研究型任务，保存报告到文件
            if "final_report" in result and result["final_report"]:
                is_action = result.get("is_action_task", False)
                if not is_action and len(result["final_report"]) > 100:
                    # 只对研究报告（非操作型）保存文件
                    report_filename = f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    report_path = os.path.join(settings.OUTPUT_DIR, report_filename)
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(result["final_report"])
                    print(f"\n💾 报告已保存: {report_path}")
            
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n⏹️ 用户中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 5. 清理资源
    print("\n🔄 正在关闭资源...")
    try:
        # 关闭 MCP 连接
        try:
            from src.mcp.client import close_mcp_client
            await close_mcp_client()
        except:
            pass
    except:
        pass
    
    # 关闭 SQLite 连接
    try:
        workflow.close()
    except:
        pass
    
    print("👋 程序结束")

if __name__ == "__main__":
    asyncio.run(main())