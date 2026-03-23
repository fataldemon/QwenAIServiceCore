import base64
from contextlib import asynccontextmanager
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoTokenizer, AutoProcessor

from llm.local_llm_manage import vllm_start_engine, chat, chat_on_setting
from models.base import (ModelCard, ModelList, ChatCompletionRequest,
                         ChatCompletionResponse)
from template import _get_args
from utils.websocketutils import WebsocketManager


class BasicAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(f"{username}:{password}".encode()).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get("Authorization")
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)
websocket_manager = WebsocketManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#
# Temporarily, the system role does not work as expected.
# We advise that you write the setups for role-play in your query,
# i.e., use the user role instead of the system role.
#
# TODO: Use real system role when the model is ready.
#
@app.post("/v1/assistant/completions", response_model=ChatCompletionResponse)
async def completion_without_lora(request: ChatCompletionRequest):
    global tokenizer

    choice_data = await chat(
        engine=engine,
        tokenizer=tokenizer,
        request=request,
        max_tokens=3000
    )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global tokenizer, engine

    choice_data = await chat_on_setting(
        engine=engine,
        tokenizer=tokenizer,
        max_tokens=600,
        request=request,
        active_lora_path=args.lora_path,
        index=0
    )

    # 向websocket连接广播数据
    await websocket_manager.broadcast(choice_data.json())
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


@app.websocket("/ws/{ws_mode}")
async def websocket_endpoint(ws_mode: str, websocket: WebSocket):  # ws_mode取值为"text"和"binary"
    global tokenizer, engine

    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json(mode=ws_mode)
            print(f"Data received: {data}")
            try:
                request = ChatCompletionRequest.parse_obj(data)

                choice_data = await chat_on_setting(
                    engine=engine,
                    tokenizer=tokenizer,
                    max_tokens=600,
                    request=request,
                    active_lora_path=args.lora_path,
                    index=1
                )

                await websocket_manager.send_message_to_client(choice_data.json(), websocket)
                print(f"Message sent: {choice_data.message.content}")
            except ValidationError as e:
                print("数据验证失败：", e.json())
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    args = _get_args()

    # tokenizer = AutoTokenizer.from_pretrained(
    tokenizer = AutoProcessor.from_pretrained(
        args.checkpoint_path,
    )

    if args.api_auth:
        app.add_middleware(
            BasicAuthMiddleware, username=args.api_auth.split(":")[0], password=args.api_auth.split(":")[1]
        )

    engine = vllm_start_engine(
        model=args.checkpoint_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1
    )

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)


