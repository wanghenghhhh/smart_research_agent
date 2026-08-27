# Smart Research Agent

基于 **LangGraph + ReAct + Adaptive RAG + MCP** 的智能科研助手。

## ✨ 核心能力

- 🔍 **深度研究**：自动规划研究路径，调用搜索/RAG，生成结构化报告
- 🧠 **自主决策**：ReAct Agent 自主判断调用哪些工具
- 🗄️ **持久记忆**：SQLite 存储跨会话对话历史
- 🛠️ **本地操作**：MCP 协议读取文件、执行命令
- 📊 **可观测性**：内置追踪和性能指标

## 🚀 快速开始

### 方式一：本地运行（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 然后编辑 .env，填入你的 OPENAI_API_KEY 和 TAVILY_API_KEY

# 3. CLI 交互模式（直接对话）
python main.py

# 4. FastAPI 服务模式（启动 API 服务）
python -m api.main
# 访问 http://localhost:8080/docs 查看 Swagger 接口文档

