import httpx
import asyncio
import logging
from typing import Optional, List, ClassVar
from dataclasses import dataclass

from client import Client
from config import settings

class EmbeddingClient(Client):

    def __init__(self,base_url:str):
        base_url = (
            base_url or f"http://{settings.llama_host}:{settings.llama_port}"
        ).rstrip("/")
        super().__init__(base_url)
        self.logger = logging.getLogger(self.__class__.__name__)

