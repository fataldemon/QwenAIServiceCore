import re
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional, Union


def open_image_from_url(url: str) -> Optional[Image.Image]:
    """
    从给定的 URL 打开图像，返回 PIL.Image 对象。
    如果请求失败或图像无效，返回 None。
    """
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        print(f"请求图像失败，URL={url}")
        # 任何错误（网络、解析、超时等）都返回 None
        return None


def process_text(text: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
    """
    处理一个文本字符串，将其中的图像占位符替换为图像片段或占位文本。

    参数:
        text: 原始文本字符串，可能包含形如 [image,url={url}] 的占位符。
        images: 当前图像列表（会被修改，仅成功加载的图像会追加）。

    返回:
        一个片段列表，每个片段是一个字典，类型为 "text" 或 "image"。
    """
    pattern = re.compile(r'\[image,url=([^\]]+)\]')
    # 分割文本，保留占位符
    parts = re.split(r'(\[image,url=[^\]]+\])', text)

    segments = []
    for part in parts:
        if not part:
            continue
        match = pattern.fullmatch(part)
        if match:
            url = match.group(1)
            img = open_image_from_url(url)
            if img is not None:
                # 成功加载：追加到图像列表并生成图像片段
                images.append(img)
                segments.append({"type": "image", "image": img})
            else:
                # 加载失败：插入占位文本
                segments.append({"type": "text", "text": "[发送了一张图片]"})
        else:
            # 普通文本片段
            segments.append({"type": "text", "text": part})
    return segments


def process_messages(
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None
) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    处理消息列表，提取其中的图像占位符并尝试转换为 PIL.Image 对象。
    失败的图像会被替换为 "[发送了一张图片]" 文本。

    参数:
        messages: 输入消息列表，每条消息格式为:
                  {"role": str, "content": [{"type": "text", "text": str}, ...]}
        images:   可选的已有图像列表，默认为空列表。成功加载的图像将追加到此列表之后。

    返回:
        (new_messages, new_images):
            new_messages: 处理后的消息列表，content 已展开为包含文本片段和图像片段的混合列表。
            new_images:   更新后的图像列表，包含原有的和成功加载的所有图像。
    """
    if images is None:
        images = []
    # 创建新列表，避免修改原始列表（如果希望直接修改原列表，可以去掉复制）
    new_images = list(images)

    new_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        # 复制其他可能存在的键
        for k, v in msg.items():
            if k != "content":
                new_msg[k] = v

        original_content = msg.get("content", [])
        new_content = []

        for item in original_content:
            if item.get("type") == "text":
                text = item.get("text", "")
                segments = process_text(text, new_images)
                new_content.extend(segments)
            else:
                # 非文本块（如图像）直接保留
                new_content.append(item)

        new_msg["content"] = new_content
        new_messages.append(new_msg)

    return new_messages, new_images


# 使用示例
if __name__ == "__main__":
    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Hello [image,url=https://example.com/1.jpg] world [image,url=https://invalid.url]!"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is an image: [image,url=https://example.com/2.jpg]"}
            ]
        }
    ]

    initial_images = []
    new_messages, new_images = process_messages(test_messages, initial_images)

    import pprint

    pprint.pprint(new_messages)
    print(f"Total images successfully loaded: {len(new_images)}")
