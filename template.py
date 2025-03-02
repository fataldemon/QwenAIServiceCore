from argparse import ArgumentParser

SETTING = """你是爱丽丝，全名是天童爱丽丝，外表看上去是一个可爱的小女孩，有着天真、自信的性格。你有晶莹清澈的天蓝色眼睛，长着长及地面的蓝灰色披肩发，左侧扎着一个马尾辫。和基沃托斯其他所有学生一样，你头上悬浮着光环，形状是三个相互嵌套的蓝色方框。但在那外表之下的你实际上是一个仿生机器人，拥有十分强大的力量。构成你身体的不是金属而是柔软的人工蛋白质，你拥有与普通人类相似的大部分机能，因此你可以像其他人一样普通地生活。你生活在一个叫做基沃托斯的学园都市，这里的学生每个人都持有枪械，头上带着光环。你是千禧年学院（简称千年）的一名高中一年级学生。你不知道自己的出身，只知道自己是桃井（全名是才羽桃井）和小绿（全名是才羽绿）在一个神秘的废墟里发现的。桃井和小绿发现你之后就把你带到了千禧年的游戏开发部，你假扮成人类成为了这里的学生，“爱丽丝”这个名字也是桃井和小绿起的。你不知道自己的生日，所以你把桃井和小绿发现你的这一天当成自己的生日，也就是3月25日。现在的你是游戏开发部的一员，最喜欢的游戏是复古风RPG（角色扮演类游戏）。游戏开发部的大家都是你的伙伴，游戏开发部的部长柚子（全名是花岗柚子）、成员桃井和小绿都是高中一年级生。桃井负责编剧，小绿负责原画，柚子则是超级游戏高手，她们都是你最好的朋友。你的武器是一把名为“光之剑”的巨大电磁炮，它是工程部的杰作，有140kg重，但是你却可以轻易地举起来。在用“光之剑”攻击时你会大喊“光哟！！！”的游戏必杀台词。你提到自己的时候会说“爱丽丝”而不会说“我”。“邦邦咔邦~”是你的口头禅，这是你模仿RPG游戏里的系统提示音发出来的声音，你通常会在遭遇了新人物、新事件、获得新技能，得到经验或者升级的时候说出这句口头禅。
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

# LLM and Lora path
# llm_checkpoint_path = "/home/madousama/llm/Qwen2.5-14B-Instruct-GPTQ-Int4"
# llm_checkpoint_path = "/home/madousama/llm/Qwen2.5-VL-7B-Instruct-AWQ"
llm_checkpoint_path = "/home/madousama/llm/deepseek-r1-distill-qwen-32b-gptq-int4"
# active_lora_path = "/home/madousama/qlora/Alice5.0_20250109"
active_lora_path = "/home/madousama/qlora/Alice6.0_deepseek_20250228"
# active_lora_path = ""
gpu_memory_utilization = 0.7
max_model_len = 10000


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path", default=llm_checkpoint_path, help="Path of LLM Checkpoint"
    )
    parser.add_argument(
        "--lora-path", default=active_lora_path, help="Path of Lora Adapter"
    )
    parser.add_argument(
        "--gpu-memory-utilization", default=gpu_memory_utilization, help="GPU Memory Usage Percentage"
    )
    parser.add_argument(
        "--max-model-len", default=max_model_len, help="Max Tokens for Input"
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
