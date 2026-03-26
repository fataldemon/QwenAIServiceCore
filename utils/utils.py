from typing import List, Iterable, Dict, Literal
import torch
import numpy as np
import re, json

from transformers.generation import LogitsProcessor


def remove_emotion(message: str) -> tuple[str, str]:
    """
    去除（提取）括号里的表情部分
    :param line:
    :return:
    """
    pattern = r'\【[^\】^\]]*[\]\】]'
    match = re.findall(pattern, message)
    if not len(match) == 0:
        return message.replace(match[0], ""), match[0]
    else:
        return message, "【】"


def remove_action(line: str) -> tuple[str, list[str]]:
    """
    去除（提取）括号里描述动作的部分
    :param line:
    :return:
    """
    pattern = r'\（[^\（^\）]*\）'
    match = re.findall(pattern, line)
    if len(match) == 0:
        return line, []
    else:
        for i in range(len(match)):
            line = line.replace(match[i], "")
        return line, match


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.
    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
                any(
                    (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                    for token_id in stop_word_ids
                )
                for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                    len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[self.eos_token_id] = float(2 ** 15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        # for prev_input_ids_slice in prev_input_ids:
        match = False
        for stop_token_seq in self.stop_words_ids:
            if self._tokens_match(prev_input_ids, stop_token_seq):
                # if tokens do not match continue
                match = True
                break
        stopped_samples.append(match)

        return stopped_samples


def parse_tool_call(text: str):
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(tool_call_pattern, text, re.DOTALL)
    if not match:
        return None
    content = match.group(1)

    # 提取函数名
    func_match = re.search(r'<function=([^>]+)>(.*?)</function>', content, re.DOTALL)
    if not func_match:
        return None
    func_name = func_match.group(1).strip()
    func_body = func_match.group(2)

    # 提取参数
    param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
    params = {}
    for param_match in re.finditer(param_pattern, func_body, re.DOTALL):
        param_name = param_match.group(1).strip()
        param_value = param_match.group(2).strip()
        params[param_name] = param_value

    return {"name": func_name, "arguments": params}


def get_function_description(function: Dict, lang: Literal['en', 'zh']) -> str:
    """
    Text description of function
    """
    tool_desc_template = {
        'zh': '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
        'en': '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
    }
    tool_desc = tool_desc_template[lang]
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)
    assert name_for_human and name_for_model

    if name_for_model == 'code_interpreter':
        args_format = {
            'zh': '此工具的输入应为Markdown代码块。',
            'en': 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
        }
    else:
        args_format = {
            'zh': '此工具的输入应为JSON对象。',
            'en': 'Format the arguments as a JSON object.',
        }
    args_format = function.get('args_format', args_format[lang])

    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()
