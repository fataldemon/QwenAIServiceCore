import re
import copy
import random
import datetime
from fastapi import HTTPException

import torch
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine, TokensPrompt
from vllm.lora.request import LoRARequest
from models.base import (ModelCard, ModelList, ChatMessage, ChatCompletionRequest,
                         ChatCompletionResponseChoice, ChatCompletionResponse)
from embedding.embedding import (process_embedding, vector_search, reorganize_index, check_emotion,
                                 write_as_memory, generate_vector)
from utils.utils import get_function_description, remove_action, remove_emotion, StopWordsLogitsProcessor
from template import SETTING, REACT_INSTRUCTION, _TEXT_COMPLETION_CMD, _get_args


def vllm_start_engine(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int
) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=model,
        device="cuda",
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=True,
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
            ac = content.rfind("Action:")  # 找到action的位置
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


# 解析ReAct格式的响应数据
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


# 调用LLMEngine进行推理
async def vllm_generate(engine: AsyncLLMEngine, tokenizer, messages: list, gen_kwargs, max_tokens,
                        active_lora_path: str, logits_processor: LogitsProcessorList = None) -> str:
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f">>>Input Tokens: {len(input_ids)} tokens")
    if logits_processor is not None:
        sampling_params = SamplingParams(
            **gen_kwargs,
            max_tokens=max_tokens,
            logits_processors=logits_processor,
        )
    else:
        sampling_params = SamplingParams(
            **gen_kwargs,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id]
        )
    timestamp = datetime.datetime.now()
    request_id = f"{timestamp.strftime("%Y%m%d%H%M%S")}{random.randint(1, 1000)}"
    # 没有Lora路径时调用原生模型
    if active_lora_path != "":
        result_generator = engine.generate(
            prompt=TokensPrompt(prompt_token_ids=input_ids),
            # inputs={"prompt_token_ids": input_ids},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=LoRARequest(lora_name="alice", lora_int_id=1, lora_path=active_lora_path)
        )
    else:
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
    print(f"{message}")
    # 调用无Lora的大模型
    response = await vllm_generate(
        engine,
        tokenizer,
        max_tokens=max_tokens,
        messages=message,
        gen_kwargs=gen_kwargs,
        active_lora_path=""
    )
    print(f"Assistant:{response}")
    # 如果是知识点概要就存储
    if type == 1:
        print(f"Knowledge Saved: {response}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        thought="",
        message=ChatMessage(role="assistant", content=response),
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
    else:
        messages = history + [{"role": "user", "content": query}]

    response = await vllm_generate(
        engine,
        tokenizer,
        max_tokens=max_tokens,
        messages=messages,
        gen_kwargs=gen_kwargs,
        logits_processor=logits_processor,
        active_lora_path=active_lora_path
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
            subject="setting",
            instruct='给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的设定信息'
        )
        embedding_list = reorganize_index(embedding_list, result_list, 20)
        choice_data.embedding_list = embedding_list
    else:
        choice_data.embedding_list = []
    return choice_data
