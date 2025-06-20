import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
import json
from config import settings 

@dataclass
class LlamaResponse:
    """Response from llama.cpp server"""
    content: str
    tokens_generated: int
    tokens_per_second: float
    completion_time: float
    model: str

class LLMClientBase(ABC):
    """Abstract base class for different LLM clients"""
    
    def __init__(self):
        self.logger = logging.getLogger("LLMClient")

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LlamaResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM server is healthy"""
        pass

class LlamaClient(LLMClientBase):
    """
    HTTP client for llama.cpp server
    Optimized for Phi-3.5-mini Q3_K_M on 4GB RAM Pi
    """
    
    def __init__(
        self,
        base_url: str = f"{settings.llama_host}:{settings.llama_port}",
        timeout: int = 180,  # 3 minutes for generation
        max_connections: int = 2,  # Limited for Pi resources
        max_tokens: int = 200,  # 4GB RAM limit
    ):
        super().__init__()
        self.logger = logging.getLogger("LLMClient.LLama")
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_tokens = max_tokens
        
        # Connection pool settings for Pi cluster
        connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        self.logger.info(f"LlamaClient initialized for {base_url}")

    def _format_phi35_prompt(self, prompt: str, system_message: str = None) -> str:
        """
        Format prompt for Phi-3.5-mini using proper chat template
        Template: <|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n
        """
        if system_message is None:
            system_message = "You are a helpful AI assistant that provides accurate and concise answers."
        
        formatted = f"<|system|>\n{system_message}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        return formatted
    

    async def generate(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = None,
        temperature: float = 0.2,
        top_p: float = 0.7,
        stop_sequences: List[str] = None
    ) -> LlamaResponse:
        """
        Generate text using llama.cpp server
        
        Args:
            prompt: User prompt/question
            system_message: System instructions (optional)
            max_tokens: Max tokens to generate (defaults to instance setting)
            temperature: Randomness (0.0-2.0)
            top_p: Nucleus sampling (0.0-1.0)
            stop_sequences: Stop generation on these strings
        
        Returns:
            LlamaResponse with generated text and metadata
        """
        if not self.session or self.session.closed:
            raise RuntimeError("LlamaClient session is closed")
        
        # Format prompt for Phi-3.5-mini
        formatted_prompt = self._format_phi35_prompt(prompt, system_message)
        
        # Prepare request payload
        payload = {
            "prompt": formatted_prompt,
            "n_predict": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,  # Non-streaming answer
            "stop": stop_sequences or ["<|end|>", "<|user|>"],
        }
        
        try:
            self.logger.info(f"Sending generation request to {self.base_url}/completion")
            
            async with self.session.post(
                f"{self.base_url}/completion",
                json=payload
        ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"LLM server error: {error_text}"
                    )
                
                result = await response.json()
                
                # Parse llama.cpp response
                return LlamaResponse(
                    content=result.get("content", "").strip(),
                    tokens_generated=result.get("tokens_predicted", 0),
                    tokens_per_second=result.get("timings", {}).get("predicted_per_second", 0.0),
                    completion_time=result.get("timings", {}).get("predicted_ms", 0.0) / 1000.0,
                    model="phi-3.5-mini"
                )
                
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP client error: {e}")
            raise
        except asyncio.TimeoutError:
            self.logger.error("Request timeout - llama.cpp server too slow")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in generate(): {e}")
            raise
    
    async def health_check(self) -> bool:
        """
        Check if llama.cpp server is responding
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    async def close(self):
        
    
# Factory function 
def create_llama_client(base_url: str = None) -> LlamaClient:
    """
    Factory function to create LlamaClient with environment-based config
    
    Args:
        base_url: Override default URL
    
    Returns:
        Configured LlamaClient instance
    """
    if base_url is None:
        base_url = f"{settings.llama_host}:{settings.llama_port}"
    
    return LlamaClient(base_url=base_url)