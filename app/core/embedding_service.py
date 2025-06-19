from typing import List


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # Track the model name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    async def generate_embedding(self, text: str) -> tuple[List[float], str]:
        """Returns (embedding, model_name)"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist(), self.model_name