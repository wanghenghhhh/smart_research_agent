from sentence_transformers import SentenceTransformer
from typing import List ,Union
import numpy as np

class EmbeddingModel:
    def __init__(self,model_name:str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        print(f"✅ 嵌入模型加载成功，维度: {self.vector_size}")

    def encode(self,texts:Union[str,List[str]])-> np.ndarray:
        if isinstance(texts,str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar = False
        )
        return embeddings