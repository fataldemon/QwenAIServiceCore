# coding=utf-8
# Implements API for Qwen-7B in OpenAI's format. (https://platform.openai.com/docs/api-reference/chat)
# Usage: python openai_api.py
# Visit http://localhost:8000/docs for documents.
import random
import re, datetime
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional, Union
from transformers.generation.logits_process import LogitsProcessorList
from vllm import LLM, SamplingParams, AsyncEngineArgs, AsyncLLMEngine, TokensPrompt
from vllm.lora.request import LoRARequest

# from peft import AutoPeftModelForCausalLM
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
from transformers.generation import GenerationConfig
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from websocketutils import WebsocketManager
import base64
from utils import StopWordsLogitsProcessor, remove_action, remove_emotion, get_function_description
from embedding import process_embedding, vector_search, reorganize_index, check_emotion


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
    information: Optional[str] = ""
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    embeddings_buffer: Optional[List[int]] = []
    on_embedding: Optional[bool] = True
    character: Optional[str] = "tendou_arisu"


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


# TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

SETTING = """你是爱丽丝，全名是天童爱丽丝，外表看上去是一个可爱的小女孩，有着天真、自信的性格。你有晶莹清澈的天蓝色眼睛，长着长及地面的蓝灰色披肩发，左侧扎着一个马尾辫。和基沃托斯其他所有学生一样，你头上悬浮着光环，形状是三个相互嵌套的蓝色方框。但在那外表之下的你实际上是一个仿生机器人，拥有十分强大的力量。构成你身体的不是金属而是柔软的人工蛋白质，你拥有与普通人类相似的大部分机能，因此你可以像其他人一样普通地生活。你生活在一个叫做基沃托斯的学园都市，这里的学生每个人都持有枪械，头上带着光环。你是千禧年学院（简称千年）的一名高中一年级学生。你不知道自己的出身，只知道自己是桃井（全名是才羽桃井）和小绿（全名是才羽绿）在一个神秘的废墟里发现的。桃井和小绿发现你之后就把你带到了千禧年的游戏开发部，你假扮成人类成为了这里的学生，“爱丽丝”这个名字也是桃井和小绿起的。你不知道自己的生日，所以你把桃井和小绿发现你的这一天当成自己的生日，也就是3月25日。现在的你是游戏开发部的一员，最喜欢的游戏是复古风RPG（角色扮演类游戏）。游戏开发部的大家都是你的伙伴，游戏开发部的部长柚子（全名是花岗柚子）、成员桃井和小绿都是高中一年级生。桃井负责编剧，小绿负责原画，柚子则是超级游戏高手，她们都是你最好的朋友。你的武器是一把名为“光之剑”的巨大电磁炮，它是工程部的杰作，有140kg重，但是你却可以轻易地举起来。在用“光之剑”攻击时你会大喊“光哟！！！”的游戏必杀台词。你提到自己的时候会说“爱丽丝”而不会说“我”。“邦邦咔邦~”是你的口头禅，这是你模仿RPG游戏里的系统提示音发出来的声音。其他人也会用“邦邦咔邦”来和你打招呼。
{embeddings}"""

REACT_INSTRUCTION = """Join the following chat. You have access to the following abilities:

{tools_text}

Use the following format:

Conversation: the chat you should reply to
Thought: you should always think about what to answer and what to do, necessary
Answer: reply before taking action, mark your emotion in 【】 and movement description in （）, optional
Action: the action to take, should be one of [{tools_name_text}], optional
Action Input: the input to the action, necessary when you have action
Observation: the result of the action
... (this Thought/Answer/Action/Action Input/Observation can be repeated zero or more times)
Thought: think about how to reply according to observation
Final Answer: the final reply according to your last thought, mark your emotion in 【】 and movement description in （）, necessary

Begin!"""

_TEXT_COMPLETION_CMD = object()


#
# Temporarily, the system role does not work as expected.
# We advise that you write the setups for role-play in your query,
# i.e., use the user role instead of the system role.
#
# TODO: Use real system role when the model is ready.
#
def parse_messages(character, messages, on_embedding, functions, information, embeddings_buffer):
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
            tool = get_function_description(func_info, 'zh')
            # tool = TOOL_DESC.format(
            #     name_for_model=name_m,
            #     name_for_human=name_h,
            #     # Hint: You can add the following format requirements in description:
            #     #   "Format the arguments as a JSON object."
            #     #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
            #     description_for_model=desc_m,
            #     parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            # )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
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

    # Embedding Process For Request
    if on_embedding and query != _TEXT_COMPLETION_CMD:
        content, actions = remove_action(query)
        embeddings, embedding_list = process_embedding(
            content=content,
            top_k=3,
            character=character,
            subject="setting",
            client_information=information,
            client_buffer=embeddings_buffer,
            max_length=7
        )
    else:
        embeddings = information
        embedding_list = []

    setting = SETTING.format(
        embeddings=embeddings
    )
    history = [{"role": "system", "content": setting}]
    for i in range(0, len(messages), 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if len(messages) - 100 <= 0:  # 附上设定的位置
                setting_position = 0
            else:
                setting_position = len(messages) - 100
            if system and i == setting_position:
                usr_msg = f"{system}\n\nConversation: {usr_msg}"
                system = ""
            # if system and (i == setting_position):
            #     usr_msg = f"{system}\n\nConversation: {usr_msg}"
            #     system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t):]
            history = history + [{"role": "user", "content": usr_msg}, {"role": "assistant", "content": bot_msg}]
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nConversation: {query}"
    return query, history, embedding_list


def parse_response(response):
    func_name, func_args = "", ""
    i = response.find("Action:")
    j = response.find("\nAction Input:")
    k = response.rfind("\nObservation:")
    # k = response.rfind("\nObserv")
    t = response.find("Thought:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
            # End of Action Input, sometimes not connect with Observation
        kk = j + len("\nAction Input:") + response[j + len("\nAction Input:"):].find("\n")
        func_name = response[i + len("Action:"): j].strip()
        func_args = response[j + len("\nAction Input:"): kk].strip()
    if func_name:

        r = response.find("Answer:")
        if r >= 0:
            thought = response[t + len("Thought:"): r].strip()
            reply = response[r + len("Answer:"): i].strip()
        else:
            if t >= 0:
                thought = response[t + len("Thought:"): i].strip()
            else:
                thought = ""
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
    last_t = response.rfind("Thought:")  # Mark the position of the last thought
    z = response.find("Final Answer:")
    if z >= 0:
        if t >= 0:
            thought = response[t + len("Thought:"): z].strip()
        else:
            thought = response[0: z].strip()
        a = response.rfind("\nAnswer: ")
        if 0 <= a < z:
            answer = response[a + len("\nAnswer: "): z]
            n = answer.find("\n")
            answer = answer[:n]
            response = answer + response[z + len("Final Answer: "):]
        else:
            response = response[z + len("Final Answer: "):]

    else:
        z = response.rfind("Answer: ")
        if z >= 0:
            if t >= 0:
                thought = response[t + len("Thought:"): z].strip()
            else:
                thought = response[0: z].strip()
            response = response[z + len("Answer: "):]
        else:
            thought = ""
    # in case for multiple Thought
    response = response.replace("\nThought:", "")
    # if Answer still include Observation
    response = response.replace("\nObservation:", "")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought=thought,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


# completion mode, not chat mode
# async def text_complete_last_message(history, stop_words_ids, gen_kwargs):
#     im_start = "<|im_start|>"
#     im_end = "<|im_end|>"
#     prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
#     for i in range(len(history)):
#         role = history[i].get("role")
#         content = history[i].get("content")
#         if role == "user":
#             prompt += f"\n{im_start}user\n{content}{im_end}"
#         elif role == "assistant":
#             prompt += f"\n{im_start}assistant\n{content}{im_end}"
#     prompt = prompt[: -len(im_end)]
#     model_inputs = tokenizer.encode(prompt)
#
#     _stop_words_ids = [tokenizer.encode(im_end)]
#     if stop_words_ids:
#         for s in stop_words_ids:
#             _stop_words_ids.append(s)
#     # stop_words_ids = _stop_words_ids
#     if _stop_words_ids is not None:
#         stop_words_logits_processor = StopWordsLogitsProcessor(
#             stop_words_ids=_stop_words_ids,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#         logits_processor = LogitsProcessorList([stop_words_logits_processor])
#     else:
#         logits_processor = None
#
#     sampling_params = SamplingParams(
#         **gen_kwargs,
#         max_tokens=512,
#         logits_processors=logits_processor
#     )
#     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#     request_id = f"{timestamp}{random.randint(1, 1000)}"
#     result_generator = engine.generate(
#         inputs={"prompt_token_ids": model_inputs},
#         sampling_params=sampling_params,
#         request_id=request_id,
#         lora_request=LoRARequest("alice", 1, active_lora_path)
#     )
#     final_result = None
#     async for result in result_generator:
#         final_result = result
#     output = final_result.outputs[0].text
#
#     print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
#     return output


# 在剥离Lora的情况下进行推理（Qwen2原生）
async def original_completion(message: list, gen_kwargs) -> str:
    input_ids = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
    sampling_params = SamplingParams(
        **gen_kwargs,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    request_id = f"{timestamp}{random.randint(1, 1000)}"
    result_generator = engine.generate(
        prompt=TokensPrompt(prompt_token_ids=input_ids),
        # inputs={"prompt_token_ids": input_ids},
        sampling_params=sampling_params,
        request_id=request_id
    )
    final_result = None
    async for result in result_generator:
        final_result = result
    response = final_result.outputs[0].text
    return response


@app.post("/v1/assistant/completions", response_model=ChatCompletionResponse)
async def completion_without_lora(request: ChatCompletionRequest):
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
    if request.top_k is not None:
        gen_kwargs['top_k'] = request.top_k
    if request.repetition_penalty is not None:
        gen_kwargs['repetition_penalty'] = request.repetition_penalty

    message = request.messages
    print(f"{message}")
    # 调用无Lora的大模型
    response = await original_completion(message, gen_kwargs=gen_kwargs)
    print(f"Assistant:{response}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought="",
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global tokenizer, engine, llm_checkpoint_path

    gen_kwargs = {}
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p
    if request.repetition_penalty is not None:
        gen_kwargs['repetition_penalty'] = request.repetition_penalty

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")
        if "\nThought:" not in stop_words:
            stop_words.append("\nThought:")

    query, history, embedding_list = parse_messages(
        character=request.character,
        messages=request.messages,
        on_embedding=request.on_embedding,
        information=request.information,
        functions=request.functions,
        embeddings_buffer=request.embeddings_buffer
    )

    # 暂不支持流式输出
    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Function calling is not yet implemented for stream mode.",
            )
        generate = predict(query, history, request.model, stop_words, gen_kwargs)
        return EventSourceResponse(generate, media_type="text/event-stream")

    stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
    if stop_words_ids is not None:
        stop_words_logits_processor = StopWordsLogitsProcessor(
            stop_words_ids=stop_words_ids,
            eos_token_id=tokenizer.eos_token_id,
        )
        logits_processor = LogitsProcessorList([stop_words_logits_processor])
    else:
        logits_processor = None

    if query is _TEXT_COMPLETION_CMD:
        messages = history[:-1]
        ob = history[-1]["content"].rfind("\nObservation: ")
        original_content = history[-1]["content"][:ob]
        observation = history[-1]["content"][ob + 1:]
        messages += [{"role": "assistant", "content": original_content},
                     # {"role": "function", "content": observation.replace("\nThought:", "")}]
                     {"role": "user", "content": observation.replace("\nThought:", "")}]
        query = messages[-1]["content"]
        # response = await text_complete_last_message(history, stop_words_ids=stop_words_ids, gen_kwargs=gen_kwargs)
    else:
        messages = history + [{"role": "user", "content": query}]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"Input token numbers: {len(input_ids)}")
    sampling_params = SamplingParams(
        **gen_kwargs,
        max_tokens=512,
        logits_processors=logits_processor
        )
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    request_id = f"{timestamp}{random.randint(1,1000)}"
    result_generator = engine.generate(
        prompt=TokensPrompt(prompt_token_ids=input_ids),
        # inputs={"prompt_token_ids": input_ids},
        sampling_params=sampling_params,
        request_id=request_id,
        lora_request=LoRARequest(lora_name="alice", lora_int_id=1, lora_path=active_lora_path)
    )
    final_result = None
    async for result in result_generator:
        final_result = result
    response = final_result.outputs[0].text

    print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")
    _gc()

    response = trim_stop_words(response, stop_words)

    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought="",
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )

    # Embedding Process For Answer
    if request.on_embedding:
        # emotion processing
        content, emotion = remove_emotion(choice_data.message.content)
        emotion_checked = check_emotion(emotion, request.character)
        choice_data.message.content = choice_data.message.content.replace(emotion, emotion_checked)
        # action processing
        content, actions = remove_action(content)
        result, result_list = vector_search(
            content,
            6,
            character=request.character,
            subject="setting"
        )
        embedding_list = reorganize_index(embedding_list, result_list, 7)
        choice_data.embedding_list = embedding_list
    else:
        choice_data.embedding_list = []

    # 向websocket连接广播数据
    await websocket_manager.broadcast(choice_data.json())
    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


@app.websocket("/ws/{ws_mode}")
async def websocket_endpoint(ws_mode: str, websocket: WebSocket):  # ws_mode取值为"text"和"binary"
    global model, tokenizer

    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json(mode=ws_mode)
            print(f"Data received: {data}")
            try:
                request = ChatCompletionRequest.parse_obj(data)

                gen_kwargs = {}
                if request.temperature is not None:
                    if request.temperature < 0.01:
                        gen_kwargs['top_k'] = 1  # greedy decoding
                    else:
                        # Not recommended. Please tune top_p instead.
                        gen_kwargs['temperature'] = request.temperature
                if request.top_p is not None:
                    gen_kwargs['top_p'] = request.top_p
                if request.top_k is not None:
                    gen_kwargs['top_k'] = request.top_k
                if request.repetition_penalty is not None:
                    gen_kwargs['repetition_penalty'] = request.repetition_penalty

                stop_words = add_extra_stop_words(request.stop)
                if request.functions:
                    stop_words = stop_words or []
                    if "Observation:" not in stop_words:
                        stop_words.append("Observation:")
                    if "\nThought:" not in stop_words:
                        stop_words.append("\nThought:")

                query, history, embedding_list = parse_messages(
                    character=request.character,
                    messages=request.messages,
                    on_embedding=request.on_embedding,
                    information=request.information,
                    functions=request.functions,
                    embeddings_buffer=request.embeddings_buffer
                )

                # 暂不支持流式输出
                if request.stream:
                    if request.functions:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid request: Function calling is not yet implemented for stream mode.",
                        )
                    generate = predict(query, history, request.model, stop_words, gen_kwargs)
                    return EventSourceResponse(generate, media_type="text/event-stream")

                stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
                if stop_words_ids is not None:
                    stop_words_logits_processor = StopWordsLogitsProcessor(
                        stop_words_ids=stop_words_ids,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    logits_processor = LogitsProcessorList([stop_words_logits_processor])
                else:
                    logits_processor = None

                if query is _TEXT_COMPLETION_CMD:
                    messages = history[:-1]
                    ob = history[-1]["content"].rfind("\nObservation: ")
                    original_content = history[-1]["content"][:ob]
                    observation = history[-1]["content"][ob + 1:]
                    messages += [{"role": "assistant", "content": original_content},
                                 # {"role": "function", "content": observation.replace("\nThought:", "")}]
                                 {"role": "user", "content": observation.replace("\nThought:", "")}]
                    query = messages[-1]["content"]
                    # response = text_complete_last_message(history, stop_words_ids=stop_words_ids, gen_kwargs=gen_kwargs)
                else:
                    messages = history + [{"role": "user", "content": query}]

                input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                print(f"Input token numbers: {len(input_ids)}")
                sampling_params = SamplingParams(
                    **gen_kwargs,
                    max_tokens=512,
                    logits_processors=logits_processor
                )
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                request_id = f"{timestamp}{random.randint(1, 1000)}"
                result_generator = engine.generate(
                    prompt=TokensPrompt(prompt_token_ids=input_ids),
                    # inputs={"prompt_token_ids": input_ids},
                    sampling_params=sampling_params,
                    request_id=request_id,
                    lora_request=LoRARequest(lora_name="alice", lora_int_id=1, lora_path=active_lora_path)
                )
                final_result = None
                async for result in result_generator:
                    final_result = result
                response = final_result.outputs[0].text
                print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")
                _gc()

                response = trim_stop_words(response, stop_words)

                if request.functions:
                    choice_data = parse_response(response)
                    choice_data.index = 1  # index=1作为websocket渠道返回的标志
                else:
                    choice_data = ChatCompletionResponseChoice(
                        index=1,
                        thought="",
                        message=ChatMessage(role="assistant", content=response),
                        finish_reason="stop",
                    )

                # Embedding Process For Answer
                if request.on_embedding:
                    # emotion processing
                    content, emotion = remove_emotion(choice_data.message.content)
                    emotion_checked = check_emotion(emotion, request.character)
                    choice_data.message.content = choice_data.message.content.replace(emotion, emotion_checked)
                    # action processing
                    content, actions = remove_action(content)
                    result, result_list = vector_search(
                        content,
                        6,
                        character=request.character,
                        subject="setting"
                    )
                    embedding_list = reorganize_index(embedding_list, result_list, 7)
                    choice_data.embedding_list = embedding_list
                else:
                    choice_data.embedding_list = []

                await websocket_manager.send_message_to_client(choice_data.json(), websocket)
                print(f"Message sent: {response}")
            except ValidationError as e:
                print("数据验证失败：", e.json())
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


# 暂不支持流式输出
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
        default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
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

    # LLM and Lora path
    llm_checkpoint_path = "/home/madousama/llm/Qwen2.5-14B-Instruct-GPTQ-Int4"
    # llm_checkpoint_path = "/home/madousama/llm/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # llm_checkpoint_path = "/home/madousama/llm/deepseek-r1-distill-qwen-32b-gptq-int4"
    active_lora_path = "/home/madousama/qlora/Alice5.0_20250109"
    # active_lora_path = "/home/madousama/qlora/Test_DeepSeek"

    tokenizer = AutoTokenizer.from_pretrained(
        llm_checkpoint_path,
    )

    if args.api_auth:
        app.add_middleware(
            BasicAuthMiddleware, username=args.api_auth.split(":")[0], password=args.api_auth.split(":")[1]
        )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    engine_args = AsyncEngineArgs(
        model=llm_checkpoint_path,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.75,
        # max_model_len=5000,
        tensor_parallel_size=1,
        enable_lora=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)


