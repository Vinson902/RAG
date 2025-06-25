import httpx
import asyncio
import logging
from typing import Optional, List, ClassVar
from dataclasses import dataclass
from config import settings


@dataclass
class LlamaResponse:
    content: str
    tokens_generated: int
    tokens_per_second: float
    completion_time: float
    model: str
    error: Optional[str] = None


class LlamaClient:
    """
    LlamaClient with internal HTTP client management
    """

    # Class-level shared client for all instances
    _shared_client: ClassVar[Optional[httpx.AsyncClient]] = None
    _client_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, base_url: str = None):
        self.base_url = (
            base_url or f"http://{settings.llama_host}:{settings.llama_port}"
        ).rstrip("/")
        self.logger = logging.getLogger("LlamaClient")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared HTTP client"""
        if self._shared_client is None or self._shared_client.is_closed:
            async with self._client_lock:
                # Double-check pattern
                if self._shared_client is None or self._shared_client.is_closed:
                    self._shared_client = httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            connect=10.0, read=180.0, write=30.0, pool=5.0
                        ),
                        limits=httpx.Limits(
                            max_keepalive_connections=5,
                            max_connections=5,
                            keepalive_expiry=30,
                        ),
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "pi-cluster-rag/1.0",
                        },
                        follow_redirects=False,
                        verify=False,
                    )
                    self.logger.info("Created shared httpx client")

        return self._shared_client

    def _format_prompt(self, prompt: str, system_message: str = None) -> str:
        """Format prompt for Phi-3.5-mini"""
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that provides accurate answers."
            )

        return f"<|system|>\n{system_message}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

    async def _retry_operation(self, operation, max_retries: int = 2):
        """Retry logic"""
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(0.1 * (2**attempt))

    async def generate(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 0.7,
        stop_sequences: List[str] = None,
    ) -> LlamaResponse:
        """Generate text using llama.cpp server"""

        async def _generate():
            client = await self._get_client()

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

            response = await client.post(f"{self.base_url}/completion", json=payload)
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
                model="phi-3.5-mini",
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
                model="phi-3.5-mini",
                error=str(e),
            )

    async def health_check(self) -> bool:
        """Check if llama.cpp server is healthy"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    @classmethod
    async def close_shared_client(cls):
        """Close the shared HTTP client (call during app shutdown)"""
        if cls._shared_client and not cls._shared_client.is_closed:
            await cls._shared_client.aclose()
            cls._shared_client = None
            logging.getLogger("LlamaClient").info("Closed shared httpx client")
