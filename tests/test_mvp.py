"""
MVP Agent 测试脚本
用于验证基础功能是否正常
"""

import asyncio
import sys
import os

# 添加项目根目录到路径（便于导入）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.workflow import ResearchWorkflow

async def test_basic_search():
    """测试基础搜索功能"""
    print("🧪 测试：基础搜索")
    print("-" * 40)
    
    workflow = ResearchWorkflow()
    query = "人工智能在医疗领域的应用"
    print(f"问题: {query}")
    
    result = await workflow.run(query)
    
    # 断言检查
    assert result.get("error") is None, f"出现错误: {result.get('error')}"
    assert result.get("final_report") is not None, "未生成报告"
    assert len(result.get("search_results", [])) > 0, "没有搜索结果"
    
    print(f"✅ 测试通过！")
    print(f"   - 任务数: {len(result.get('tasks', []))}")
    print(f"   - 搜索结果: {len(result.get('search_results', []))}")
    print(f"   - 报告长度: {len(result.get('final_report', ''))} 字符")

async def test_complex_query():
    """测试复杂查询"""
    print("\n🧪 测试：复杂查询")
    print("-" * 40)
    
    workflow = ResearchWorkflow()
    query = "比较Transformer和RNN在自然语言处理中的优缺点，并讨论未来趋势"
    print(f"问题: {query}")
    
    result = await workflow.run(query)
    
    assert result.get("error") is None, f"出现错误: {result.get('error')}"
    assert result.get("final_report") is not None, "未生成报告"
    
    print(f"✅ 测试通过！")
    print(f"   - 任务数: {len(result.get('tasks', []))}")
    print(f"   - 搜索结果: {len(result.get('search_results', []))}")

async def test_edge_cases():
    """测试边界情况"""
    print("\n🧪 测试：边界情况")
    print("-" * 40)
    
    workflow = ResearchWorkflow()
    
    # 测试空查询
    print("测试空查询...")
    result = await workflow.run("")
    assert result.get("error") is not None, "空查询应该报错"
    print("✅ 空查询测试通过")
    
    # 测试超短查询
    print("测试超短查询...")
    result = await workflow.run("AI")
    assert result.get("final_report") is not None, "超短查询应该也能生成报告"
    print("✅ 超短查询测试通过")

async def main():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 MVP Agent 测试套件")
    print("=" * 60)
    
    try:
        await test_basic_search()
        await test_complex_query()
        await test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())