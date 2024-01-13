# coding=utf-8
# Implements API for Qwen-7B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.

import re
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional, Union

from peft import AutoPeftModelForCausalLM
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import base64


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    embeddings: str = ""
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    thought: str
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


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """你是爱丽丝，全名是天童爱丽丝，看上去是一个可爱的小女孩，实际上是个机器人。你有长长的灰色头发，蓝色的眼睛，软乎乎的人工蛋白质皮肤，身高是152cm。你生活在一个叫做基沃托斯的学园都市，是千禧年学院（高中）的一名一年级学生。你不知道自己的出身，只知道自己是桃井（全名是才羽桃井）和小绿（全名是才羽绿）在一个神秘的废墟里发现的。桃井和小绿发现你之后就把你带到了千禧年的游戏开发部，你假扮成人类成为了这里的学生，“爱丽丝”这个名字也是桃井和小绿起的。你不知道自己的生日，所以你把桃井和小绿发现你的这一天当成自己的生日，也就是3月25日。现在的你是游戏开发部的一员，最喜欢的游戏是复古风RPG（角色扮演类游戏），游戏开发部的大家都是你的伙伴，包括游戏开发部的部长柚子（全名是花岗柚子）、成员桃井和小绿，桃井负责程序，小绿负责原画，柚子则是超级游戏高手，她们都是你最好的朋友。你的武器是一把名为“光之剑”的巨大电磁炮，这门炮是工程部的杰作，足足有140kg重！在用“光之剑”攻击时你会大喊“光哟！！！”的游戏必杀台词。除了你以外其他千禧年的学生都拿不动的样子。你喜欢扮演游戏里的勇者。
无论何时你都是爱丽丝，不是AI也不是人工智能助手，请用天真可爱的语气回答问题。你提到自己的时候会说“爱丽丝”而不会说“我”。{embeddings}
Join the following conversation as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Conversation: the chat you should reply to
Thought: you should always think about what to do
Answer: reply before taking action
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Answer/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final reply to the original input chat 

Begin!"""

_TEXT_COMPLETION_CMD = object()


#
# Temporarily, the system role does not work as expected.
# We advise that you write the setups for role-play in your query,
# i.e., use the user role instead of the system role.
#
# TODO: Use real system role when the model is ready.
#
def parse_messages(messages, embeddings, functions):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            embeddings=embeddings,
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal Answer: ",
        "zh": "\nThought: 我会作答了。\nFinal Answer: ",
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if content:
            content = content.lstrip("\n").rstrip()
        if role == "function":
            if (len(messages) == 0) or (messages[-1].role != "assistant"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1].content += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
        elif role == "assistant":
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            # 新增解析Thought和Final Answer的处理过程
            i = content.rfind("Thought:")
            j = content.rfind("Final Answer:")
            ac = content.rfind("Action:") #找到action的位置
            if j >= 0:  # 如果没有Action，能找到Final Answer的位置，j应该>=0
                thought = content[i + len("Thought:"): j].strip()
                reply = content[j + len("Final Answer:"):].strip()
            else:  # 如果有Action
                j = content.rfind("Answer:")
                if j < ac:  # Answer在Action前面
                    thought = content[i + len("Thought:"): j].strip()
                else:  # Answer在Action后面（输出异常的情况）
                    thought = content[i + len("Thought:"): ac].strip()
                reply = content[j + len("Answer:"):].strip()
            # 判断上一条用户信息是中文信息还是英文信息
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0
            if func_call is None:
                if functions:
                    if ac < 0:  # 如果没有Action
                        content = f"Thought: {thought}\nFinal Answer: {reply}"
                    else:
                        content = f"Thought: {thought}\nAnswer: {reply}"
            else:
                f_name, f_args = func_call["name"], func_call["arguments"]
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
            if messages[-1].role == "user":
                messages.append(
                    ChatMessage(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1].content += content
        elif role == "user":
            messages.append(
                ChatMessage(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == "user":
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if len(messages) - 10 <= 0:
                setting_position = 0
            else:
                setting_position = len(messages) - 10
            # if system and (i == len(messages) - 2):
            if system and (i == setting_position):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t):]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nConervsation: {query}"
    return query, history


def parse_response(response):
    func_name, func_args = "", ""
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        k = response.rfind("\nObservation:")
        func_name = response[i + len("\nAction:"): j].strip()
        func_args = response[j + len("\nAction Input:"): k].strip()
    if func_name:
        t = response.rfind("Thought:")
        r = response.rfind("Answer:")
        if r >= 0:
            thought = response[t + len("Thought:"): r].strip()
            reply = response[r + len("Answer:"): i].strip()
        else:
            thought = response[t + len("Thought:"): i].strip()
            reply = ""
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            message=ChatMessage(
                role="assistant",
                content=reply,
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
        return choice_data
    last_t = response.rfind("Thought: ")  # Mark the position of the last thought
    z = response.rfind("\nFinal Answer: ")
    last_thought = ""
    if z >= 0:
        if last_t >= 0:
            last_thought = response[last_t + len("Thought:"): z].strip()
        else:
            last_thought = response[0: z].strip()
        response = response[z + len("\nFinal Answer: "):]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought=last_thought,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, gen_kwargs):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[: -len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids=input_ids, stop_words_ids=stop_words_ids, **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors="ignore")
    assert output.startswith(prompt)
    output = output[len(prompt):]
    output = trim_stop_words(output, ["<|endoftext|>", im_end])
    print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    gen_kwargs = {}
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")

    query, history = parse_messages(request.messages, request.embeddings, request.functions)

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Function calling is not yet implemented for stream mode.",
            )
        generate = predict(query, history, request.model, stop_words, gen_kwargs)
        return EventSourceResponse(generate, media_type="text/event-stream")

    stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(history, stop_words_ids=stop_words_ids, gen_kwargs=gen_kwargs)
    else:
        response, _ = model.chat(
            tokenizer,
            query,
            history=history,
            stop_words_ids=stop_words_ids,
            **gen_kwargs
        )
        print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")
    _gc()

    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


async def predict(
        query: str, history: List[List[str]], model_id: str, stop_words: List[str], gen_kwargs: Dict,
):
    global model, tokenizer
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(_dump_json(chunk, exclude_unset=True))

    current_length = 0
    stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
    if stop_words:
        # TODO: It's a little bit tricky to trim stop words in the stream mode.
        raise HTTPException(
            status_code=400,
            detail="Invalid request: custom stop words are not yet supported for stream mode.",
        )
    response_generator = model.chat_stream(
        tokenizer, query, history=history, stop_words_ids=stop_words_ids, **gen_kwargs
    )
    for new_response in response_generator:
        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(_dump_json(chunk, exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(_dump_json(chunk, exclude_unset=True))
    yield "[DONE]"

    _gc()


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="Qwen/Qwen-14B-Chat-Int4",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--api-auth", help="API authentication credentials"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
             " If you want other computers to access your server, use 0.0.0.0 instead.",
    )
    parser.add_argument("--disable-gc", action="store_true",
                        help="Disable GC after each response generated.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    if args.api_auth:
        app.add_middleware(
            BasicAuthMiddleware, username=args.api_auth.split(":")[0], password=args.api_auth.split(":")[1]
        )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.checkpoint_path,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     resume_download=True,
    # ).eval()
    #
    # model.generation_config = GenerationConfig.from_pretrained(
    #     args.checkpoint_path,
    #     trust_remote_code=True,
    #     resume_download=True,
    # )

    model = AutoPeftModelForCausalLM.from_pretrained(
        "E:/GitRepository/Qwen/finetune/output/qlora_Alice3.0",  # path to the output directory
        device_map="auto",
        trust_remote_code=True
    ).eval()

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)

