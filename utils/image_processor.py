import re
import os
import hashlib
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional

# 缓存目录（项目根目录下）
CACHE_DIR = "images_cache"


def ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def open_image_from_url(url: str) -> Optional[Image.Image]:
    """
    从 URL 获取图像，优先使用本地缓存。
    成功返回 PIL.Image 对象，失败返回 None。
    """
    ensure_cache_dir()

    # 计算 URL 的 MD5 作为缓存文件名
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash)

    # 尝试从缓存读取
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                img_data = f.read()
            img = Image.open(BytesIO(img_data))
            return img
        except Exception:
            # 缓存文件损坏，删除后重新下载
            try:
                os.remove(cache_path)
            except OSError:
                pass  # 删除失败则继续下载

    # 缓存不存在或损坏，执行下载
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=5)
        response.raise_for_status()
        img_data = response.content

        # 检查是否为有效图像（通过尝试打开）
        img = Image.open(BytesIO(img_data))

        # 保存到缓存
        try:
            with open(cache_path, 'wb') as f:
                f.write(img_data)
        except Exception:
            # 缓存写入失败不影响结果，只记录（不抛出异常）
            pass

        return img
    except Exception:
        # 任何下载或图像解析错误都返回 None
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
                images.append(img)
                segments.append({"type": "image", "image": img})
            else:
                segments.append({"type": "text", "text": "[发送了一张图片]"})
        else:
            segments.append({"type": "text", "text": part})
    return segments


def process_messages(
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None
) -> tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    处理消息列表，提取其中的图像占位符并尝试转换为 PIL.Image 对象。
    失败的图像会被替换为 "[发送了一张图片]" 文本。
    已成功加载的图像会使用本地缓存，避免重复下载。

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
    new_images = list(images)

    new_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
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
                 "text": "（老师对爱丽丝说）我发了几张图片"
                         "[image,url=https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=EhQhmeQBFNO8tUpC0cNpPDKAll3NiBjNgQwg_wooqqOj3pS9kwMyBHByb2RQgL2jAVoQ-Oa8-P12r7-yDEL60siT3noCsz2CAQJneg&spec=0&rkey=CAMSML24x5qpVNQXhWWEl7S6nk4BFUK_OHfDqmjMcOEvwo5isHUbMLuZgVx2RS5fLCQJGQ]"
                         "[image,url=https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=EhSGsn6_v0d_AfOa6jnmSGk6YNqkmRix0Qsg_woo4-em3pS9kwMyBHByb2RQgL2jAVoQId1LlJqkK6gzdofOOQZu13oCBvuCAQJneg&spec=0&rkey=CAMSML24x5qpVNQXhWWEl7S6nk4BFUK_OHfDqmjMcOEvwo5isHUbMLuZgVx2RS5fLCQJGQ]"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is another image: [image,url=https://cdnimg-v2.gamekee.com/wiki2.0/images/w_1886/h_2366/829/399789/2026/1/26/809952.png]"}
            ]
        }
    ]

    initial_images = []
    new_messages, new_images = process_messages(test_messages, initial_images)

    import pprint

    pprint.pprint(new_messages)
    print(f"Total images successfully loaded: {len(new_images)}")
