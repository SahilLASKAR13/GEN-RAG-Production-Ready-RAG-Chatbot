from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]