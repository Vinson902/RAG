from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional, Dict, Any
import httpx
import asyncio
import logging


class Client(ABC):
    """Abstract base class for all service clients"""
    def __init__(
       self, 
       service_url: str,
       timeout: float,
       max_retries: int
    ):
       self.service_url = service_url.rstrip('/')
       self.timeout = timeout
       self.max_retries = max_retries
       self._client: Optional[httpx.AsyncClient] = None
       self.logger = logging.getLogger(self.__class__.__name__)
   
    async def initialise(self) -> None:
        """Initialize HTTP client and verify service"""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self.logger.info(f"Client initialized for {self.service_url}")
        
   
    async def close(self) -> None:
       """Close HTTP client"""
       if self._client:
           await self._client.aclose()
           self._client = None
           self.logger.info(f"Client closed")
   
    async def _retry_operation(self, operation: Callable[[], Awaitable[Any]], max_retries: int = 2) -> Any:
        """Retry logic"""
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(0.1 * (2**attempt))
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initialize() first.")
        return self._client
