import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class MultimodalContent(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[List[Dict]]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[Dict]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    information: Optional[str] = ""
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = True
    stop: Optional[List[str]] = None
    embeddings_buffer: Optional[List[int]] = []
    on_embedding: Optional[bool] = True
    character: Optional[str] = "tendou_arisu"
    type: Optional[int] = 0  # 0是普通对话，1是总结知识点，2是对话历史长期记忆


class ChatCompletionResponseChoice(BaseModel):
    index: int
    thought: Optional[str]
    embedding_list: Optional[List[int]] = []
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
