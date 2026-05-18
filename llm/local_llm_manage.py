import json
import os
import random
import datetime
from fastapi import HTTPException

import torch
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine, TokensPrompt
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.lora.request import LoRARequest
from models.base import (ModelCard, ModelList, ChatMessage, ChatCompletionRequest,
                         ChatCompletionResponseChoice, ChatCompletionResponse)
from embedding.embedding import (process_embedding, vector_search, reorganize_index, check_emotion,
                                 add_knowledge)
from utils.utils import parse_tool_call, remove_action, remove_emotion, remove_trailing_hint
from utils.image_processor import process_messages
# NOTE: This module is *dead code* preserved only for documentation; it used
# to bootstrap an in-process vLLM ``AsyncLLMEngine``. The legacy module-level
# persona strings (SETTING/REPLY_INSTRUCTION/IMAGE_SETTING) have been moved
# to ``embedding/<character>/persona.json`` and are no longer exported from
# :mod:`template`. We alias to the LEGACY_* seed material so this file
# still parses if someone imports it for reference.
from template import (
    LEGACY_ALICE_SETTING as SETTING,
    LEGACY_ALICE_REPLY_INSTRUCTION as REPLY_INSTRUCTION,
    LEGACY_ALICE_IMAGE_SETTING as IMAGE_SETTING,
    _TEXT_COMPLETION_CMD,
    _get_args,
)


def vllm_start_engine(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        enable_lora: False
) -> AsyncLLMEngine:
    if not enable_lora:
        engine_args = AsyncEngineArgs(
            model=model,
            trust_remote_code=True,
            disable_log_stats=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enable_sleep_mode=True,
            enable_chunked_prefill=True,
        )
    else:
        engine_args = AsyncEngineArgs(
            model=model,
            trust_remote_code=True,
            disable_log_stats=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=True,
            enable_sleep_mode=True,
            enable_chunked_prefill=True,
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
def parse_messages(character, messages, on_embedding, information, embeddings_buffer, images):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )
    query = messages[-1].content

    # Embedding Process For Request
    if on_embedding and query != _TEXT_COMPLETION_CMD:
        content, actions = remove_action(query)
        embeddings, embedding_list = process_embedding(
            content=remove_trailing_hint(content),
            top_k=3,
            character=character,
            client_buffer=embeddings_buffer,
            max_length=7,
            client_information=""
        )
    else:
        embeddings = ""
        embedding_list = []

    setting = SETTING.format(
        embeddings=information + embeddings
    )
    system = setting + REPLY_INSTRUCTION
    history = [{"role": "system", "content": system},
               {"role": "user", "content": f"{IMAGE_SETTING}"},
               {"role": "user", "content": "----------------------CONVERSATION START FROM HERE------------------------------"}]
    for message in messages[:-1]:
        if message.role != "function":
            history.append({"role": message.role, "content": message.content})
        else:
            history.append({"role": "tool", "content": message.content})
    history, images = process_messages(history, images)
    return query, history, embedding_list, images


# 解析响应数据
def parse_response(response, finish_reason):
    if "</think>\n" in response:
        resp_messages = response.split("</think>\n")
        thought = resp_messages[0].replace("<think>\n", "")
        response = resp_messages[1]
    else:
        thought = ""
    # 工具调用处理
    function_call = parse_tool_call(response)
    if function_call is not None:
        tool_token = response.rfind("<tool_call>\n")
        if tool_token == 0:
            response = ""
        else:
            response = response[:tool_token]
        func_name = function_call.get("name")
        func_args = json.dumps(function_call.get("arguments"))
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            message=ChatMessage(
                role="assistant",
                content=response,
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            thought=thought,
            message=ChatMessage(role="assistant", content=response),
            finish_reason=finish_reason,
        )
    return choice_data


# 调用LLMEngine进行推理
async def vllm_generate(engine: AsyncLLMEngine, autoProcessor, messages: list[dict], gen_kwargs, max_tokens,
                        active_lora_path: str, tools=None, images=None, enable_thinking=True, request_id="") -> tuple[str, int, str]:
    if tools is None:
        tools = []

    # 处理多模态输入
    input_texts = autoProcessor.apply_chat_template(
        messages,
        tokenize=False,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True
    )

    # 构建 vLLM 输入
    prompt = {"prompt": input_texts}
    if images:
        prompt["multi_modal_data"] = {"image": images}

    sampling_params = SamplingParams(
        **gen_kwargs,
        max_tokens=max_tokens,
        # stop_token_ids=[autoProcessor.eos_token_id]
    )
    timestamp = datetime.datetime.now()
    if request_id == "":
        request_id = f"{timestamp.strftime("%Y%m%d%H%M%S")}{random.randint(1, 1000)}"

    # 没有Lora路径时调用原生模型
    if active_lora_path != "":
        result_generator = engine.generate(
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            prompt=prompt,
            # inputs={"prompt_token_ids": input_ids},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=LoRARequest(lora_name="alice", lora_int_id=1, lora_path=active_lora_path),
        )
    else:
        result_generator = engine.generate(
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            prompt=prompt,
            # prompt=TokensPrompt(prompt_token_ids=input_ids),
            # inputs={"prompt_token_ids": input_ids},
            sampling_params=sampling_params,
            request_id=request_id
        )
    final_result = None
    async for result in result_generator:
        final_result = result
    response = final_result.outputs[0].text
    finish_reason = final_result.outputs[0].finish_reason

    print(f"{response}\n</chat>")
    # 计算吞吐量
    time_cost = (datetime.datetime.now() - timestamp).total_seconds()
    out_tokens = len(final_result.outputs[0].token_ids)
    speed = 0
    if time_cost != 0:
        speed = out_tokens / time_cost
    print(f">>>Input Text Tokens: {len(final_result.prompt_token_ids)} tokens")
    print(
        f">>>Output token numbers: {out_tokens} tokens, Time Cost: {time_cost} s, Average Throughput: {speed} tokens/s")

    return response, out_tokens, finish_reason


async def chat(engine: AsyncLLMEngine, autoProcessor, request: ChatCompletionRequest,
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
    if request.presence_penalty is not None:
        gen_kwargs['presence_penalty'] = request.presence_penalty

    message = request.messages
    tools = request.functions
    print(f"{message}")
    # 图像存储区
    images = []
    message_formatted, images = process_messages(
        messages=[{"role": message[0].role, "content": message[0].content}],
        images=images
    )
    if type == 0:
        enable_thinking = True
    else:
        enable_thinking = False
    # 调用无Lora的大模型
    response, out_tokens, finish_reason = await vllm_generate(
        engine,
        autoProcessor,
        tools=tools,
        max_tokens=max_tokens,
        messages=message_formatted,
        gen_kwargs=gen_kwargs,
        active_lora_path="",
        enable_thinking=enable_thinking
    )
    _gc()

    print(f"Assistant:{response}")
    # 如果是知识点概要就存储
    print(f'Assistant Type: {request.type}')
    reply = response
    if "</think>\n" in response:
        resp_messages = response.split("</think>\n")
        reply = resp_messages[1].strip()
    if request.type == 1:
        add_knowledge(content=reply, character=request.character)
        print(f"Knowledge Saved: {response}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought="",
        message=ChatMessage(role="assistant", content=reply),
        finish_reason=finish_reason,
    )
    return choice_data


async def chat_on_setting(engine: AsyncLLMEngine, autoProcessor, request: ChatCompletionRequest, max_tokens: int,
                          active_lora_path: str, index: int) -> ChatCompletionResponseChoice:
    #
    if request.abort_id is not None:
        await engine.abort(request_id=request.abort_id)

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
    if request.presence_penalty is not None:
        gen_kwargs['presence_penalty'] = request.presence_penalty
    print(f">>>Tools to Call: {request.functions}")

    stop_words = add_extra_stop_words(request.stop)

    # 图像存储区
    images = []
    query, history, embedding_list, images = parse_messages(
        character=request.character,
        messages=request.messages,
        on_embedding=request.on_embedding,
        information=request.information,
        embeddings_buffer=request.embeddings_buffer,
        images=images
    )

    message_formatted, images = process_messages(
        messages=[{"role": "user", "content": query}],
        images=images
    )
    messages = history + message_formatted
    print(f"<chat>\n{history}\n{query}\n<!-- *** -->")

    response, out_tokens, finish_reason = await vllm_generate(
        engine,
        autoProcessor,
        max_tokens=max_tokens,
        messages=messages,
        gen_kwargs=gen_kwargs,
        active_lora_path=active_lora_path,
        tools=request.functions,
        images=images,
        enable_thinking=request.enable_thinking,
        request_id=request.request_id
    )

    _gc()

    response = trim_stop_words(response, stop_words)

    if out_tokens >= max_tokens:
        choice_data = ChatCompletionResponseChoice(
            index=index,
            thought=response,
            message=ChatMessage(role="assistant", content="【思考】（在自己的思绪中遨游，逐渐分了神......）[SILENCE]"),
            finish_reason="overthink",
        )
    elif request.functions:
        choice_data = parse_response(response, finish_reason)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=index,
            thought="",
            message=ChatMessage(role="assistant", content=response),
            finish_reason=finish_reason,
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
            subject="setting",
            instruct='给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的设定信息'
        )
        embedding_list = reorganize_index(embedding_list, result_list, 20)
        choice_data.embedding_list = embedding_list
    else:
        choice_data.embedding_list = []
    return choice_data
