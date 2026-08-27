import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.8-max")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 5))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_knowledge")
    VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))
settings = Settings()
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)#确保输出目录存在