import httpx
import asyncio
import logging
from typing import Optional, List, ClassVar
from dataclasses import dataclass
from config import settings
from client import Client 


@dataclass
class LlamaResponse:
    content: str
    tokens_generated: int
    tokens_per_second: float
    completion_time: float
    model: str
    error: Optional[str] = None


class LlamaClient(Client):
    """
    LlamaClient with internal HTTP client management
    """
    def __init__(
        self, 
        base_url: str, 
        timeout: float = 30.0, 
        max_retries: int = 3
    ):
        """
        Initialize embedding client.
        
        Args:
            service_url: URL of embedding service
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum retry attempts (default: 3)
        """
        base_url = (
            base_url or f"http://{settings.llama_host}:{settings.llama_port}"
        ).rstrip("/")
        super().__init__(base_url, timeout, max_retries)

    def _format_prompt(self, prompt: str, system_message: str = "") -> str:
        """Format prompt for Phi-3.5-mini"""
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that provides accurate answers."
            )

        return f"<|system|>\n{system_message}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    async def generate(
        self,
        prompt: str,
        system_message: str = "",
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.7,
        stop_sequences: List[str] = [],
    ) -> LlamaResponse:
        """Generate text using llama.cpp server"""
        await self.initialise()
        async def _generate():

            formatted_prompt = self._format_prompt(prompt, system_message)

            payload = {
                "prompt": formatted_prompt,
                "n_predict": max_tokens,
                "temperature": max(0.0, min(2.0, temperature)),
                "top_p": max(0.0, min(1.0, top_p)),
                "stream": False,
                "stop": stop_sequences or ["<|end|>", "<|user|>", "<|system|>"],
                "n_keep": -1,
                "repeat_penalty": 1.1,
                "mirostat": 0,
            }

            response = await self.client.post("/completion", json=payload)
            response.raise_for_status()

            result = response.json()

            # Clean response
            content = result.get("content", "").strip()
            for token in ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
                content = content.replace(token, "").strip()

            timings = result.get("timings", {})

            return LlamaResponse(
                content=content,
                tokens_generated=result.get("tokens_predicted", 0),
                tokens_per_second=timings.get("predicted_per_second", 0.0),
                completion_time=timings.get("predicted_ms", 0.0) / 1000.0,
                model=settings.model, # llama unable to pull model name from gguf 
            )

        try:
            return await self._retry_operation(_generate)
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return LlamaResponse(
                content="",
                tokens_generated=0,
                tokens_per_second=0.0,
                completion_time=0.0,
                model=settings.model,
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if llama.cpp server is healthy"""
        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False