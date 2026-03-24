import json
import os
import random
import datetime
from fastapi import HTTPException

import torch
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine, TokensPrompt
from vllm.lora.request import LoRARequest
from models.base import (ModelCard, ModelList, ChatMessage, ChatCompletionRequest,
                         ChatCompletionResponseChoice, ChatCompletionResponse)
from embedding.embedding import (process_embedding, vector_search, reorganize_index, check_emotion,
                                 add_knowledge)
from utils.utils import get_function_description, remove_action, remove_emotion, StopWordsLogitsProcessor
from utils.image_processor import process_message
from template import SETTING, REPLY_INSTRUCTION, IMAGE_SETTING, _TEXT_COMPLETION_CMD, _get_args
from PIL import Image


def vllm_start_engine(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int
) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=model,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        # enable_lora=True,
        enable_sleep_mode=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


def _gc(forced: bool = False):
    args = _get_args()
    if args.disable_gc and not forced:
        return

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


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


# 解析ReAct格式的请求数据
def parse_messages(character, messages, on_embedding, information, embeddings_buffer):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )
    query = ""
    for content in messages[-1].content:
        if content.get("type") == "text":
            query += content.get("text")

    # Embedding Process For Request
    if on_embedding and query != _TEXT_COMPLETION_CMD:
        content, actions = remove_action(query)
        embeddings, embedding_list = process_embedding(
            content=content,
            top_k=3,
            character=character,
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
    system = setting + REPLY_INSTRUCTION
    history = [
        {"role": "system", "content": [{"type": "text", "text": system}]}
    ]
    for message in messages[:-1]:
        history.append({"role": message.role, "content": [{"type": "text", "text": message.content}]})
    return query, history, embedding_list


# 解析响应数据
def parse_response(response):
    if "</think>\n" in response:
        resp_messages = response.split("</think>\n")
        thought = resp_messages[0].replace("<think>\n", "")
        response = resp_messages[1]
    else:
        thought = ""
    if "\n\n<tool_call>\n" and "\n</tool_call>" in response:
        tool_token = response.rfind("<tool_call>\n")
        rev_tool_token = response.rfind("</tool_call>")
        tool_call = response[tool_token + len("<tool_call>\n"):rev_tool_token]
        if tool_token == 0:
            response = ""
        else:
            response = response[:tool_token]
        tool_json = json.loads(tool_call)
        func_name = tool_json.get("name")
        func_args = json.dumps(tool_json.get("arguments"))
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            message=ChatMessage(
                role="assistant",
                content=[{"type": "text", "text": response}],
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            message=ChatMessage(role="assistant", content=[{"type": "text", "text": response}]),
            finish_reason="stop",
        )
    return choice_data


# 调用LLMEngine进行推理
async def vllm_generate(engine: AsyncLLMEngine, tokenizer, messages: list, gen_kwargs, max_tokens,
                        active_lora_path: str, tools=None) -> str:
    if tools is None:
        tools = []
    # input_ids = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     tools=tools,
    #     add_generation_prompt=True,
    #     enable_thinking=True
    # )
    # 处理多模态输入
    processed = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,  # 关键：让处理器返回字典，包含所有必要字段
    )

    # 提取 token ids 和多模态数据
    input_ids = processed["input_ids"][0]
    pixel_values = processed.get("pixel_values")  # 可能需要转换为 tensor

    # 构建 vLLM 输入
    inputs = {
        "prompt_token_ids": input_ids,
    }
    if pixel_values is not None:
        inputs["multi_modal_data"] = {"image": pixel_values}  # 根据模型调整键名

    print(f">>>Input Tokens: {len(input_ids)} tokens")
    sampling_params = SamplingParams(
        **gen_kwargs,
        max_tokens=max_tokens,
        # stop_token_ids=[tokenizer.eos_token_id]
    )
    timestamp = datetime.datetime.now()
    request_id = f"{timestamp.strftime("%Y%m%d%H%M%S")}{random.randint(1, 1000)}"
    # 没有Lora路径时调用原生模型
    if active_lora_path != "":
        result_generator = engine.generate(
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            prompt=inputs,
            # inputs={"prompt_token_ids": input_ids},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=LoRARequest(lora_name="alice", lora_int_id=1, lora_path=active_lora_path),
        )
    else:
        result_generator = engine.generate(
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            prompt=inputs,
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            # inputs={"prompt_token_ids": input_ids},
            sampling_params=sampling_params,
            request_id=request_id
        )
    final_result = None
    async for result in result_generator:
        final_result = result
    response = final_result.outputs[0].text
    # 计算吞吐量
    time_cost = (datetime.datetime.now() - timestamp).total_seconds()
    out_tokens = len(final_result.outputs[0].token_ids)
    speed = 0
    if time_cost != 0:
        speed = out_tokens / time_cost
    print(
        f">>>Output token numbers: {out_tokens} tokens, Time Cost: {time_cost} s, Average Throughput: {speed} tokens/s")

    return response


async def chat(engine: AsyncLLMEngine, tokenizer, request: ChatCompletionRequest,
               max_tokens: int) -> ChatCompletionResponseChoice:
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
    tools = request.functions
    print(f"{message}")
    # 调用无Lora的大模型
    response = await vllm_generate(
        engine,
        tokenizer,
        tools=tools,
        max_tokens=max_tokens,
        messages=message,
        gen_kwargs=gen_kwargs,
        active_lora_path=""
    )
    print(f"Assistant:{response}")
    # 如果是知识点概要就存储
    print(f'Assistant Type: {request.type}')
    if request.type == 1:
        reply = response
        if "<think>" in response and "</think>" in response:
            index_t = response.rfind("</think>\n\n")
            if index_t != -1:
                reply = response[index_t + len("</think>\n\n"):]
        add_knowledge(content=reply, character=request.character)
        print(f"Knowledge Saved: {response}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought="",
        message=ChatMessage(role="assistant", content=[{"type": "text", "text": response}]),
        finish_reason="stop",
    )
    return choice_data


async def chat_on_setting(engine: AsyncLLMEngine, tokenizer, request: ChatCompletionRequest, max_tokens: int,
                          active_lora_path: str, index: int) -> ChatCompletionResponseChoice:
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
    print(f">>>Tools to Call: {request.functions}")

    stop_words = add_extra_stop_words(request.stop)

    query, history, embedding_list = parse_messages(
        character=request.character,
        messages=request.messages,
        on_embedding=request.on_embedding,
        information=request.information,
        embeddings_buffer=request.embeddings_buffer
    )

    messages = history + [{"role": "user", "content": [{"type": "text", "text": query}]}]

    response = await vllm_generate(
        engine,
        tokenizer,
        max_tokens=max_tokens,
        messages=messages,
        gen_kwargs=gen_kwargs,
        active_lora_path=active_lora_path,
        tools=request.functions
    )

    print(f"<chat>\n{history}\n{query}\n<!-- *** -->\n{response}\n</chat>")
    _gc()

    response = trim_stop_words(response, stop_words)

    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=index,
            thought="",
            message=ChatMessage(role="assistant", content=[{"type": "text", "text": response}]),
            finish_reason="stop",
        )

    # Embedding Process For Answer
    if request.on_embedding:
        # emotion processing
        content, emotion = remove_emotion(choice_data.message.content[0].get("text"))
        emotion_checked = check_emotion(emotion, request.character)
        choice_data.message.content[0]["text"] = choice_data.message.content[0]["text"].replace(emotion, emotion_checked)
        # action processing
        content, actions = remove_action(content)
        result, result_list = vector_search(
            content,
            6,
            character=request.character,
            subject="setting",
            instruct='给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的设定信息'
        )
        embedding_list = reorganize_index(embedding_list, result_list, 20)
        choice_data.embedding_list = embedding_list
    else:
        choice_data.embedding_list = []
    return choice_data
