from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        
        """Load the sentence transformer model at startup"""
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Get dimensions with test embedding
            test_embedding = self.model.encode("test")
            self.dimensions = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Dimensions: {self.dimensions}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")
    
    def encode_text(self, text: str) -> List[float]:
        """Convert single text to embedding"""
        try:
            # Clean the text
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Generate embedding
            embedding = self.model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Convert multiple texts to embeddings"""
        try:
            # Clean texts
            cleaned_texts = [text.strip() for text in texts if text.strip()]
            
            if not cleaned_texts:
                raise ValueError("No valid texts provided")
            
            # Generate embeddings in batch
            embeddings = self.model.encode(cleaned_texts)
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "dimensions": self.dimensions
        }