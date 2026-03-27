import re
import os
import hashlib
import requests
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional

# 缓存目录（项目根目录下）
CACHE_DIR = "images_cache"
# 本地图片根目录（相对于项目根目录）
LOCAL_IMAGE_DIR = os.path.join("embedding", "tendou_arisu", "image")


def ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def open_image_from_local(filename: str) -> Optional[Image.Image]:
    """
    从本地固定目录读取图片。
    :param filename: 文件名（例如 "example.jpg"）
    :return: PIL.Image 对象，失败返回 None
    """
    # 构建完整路径
    file_path = os.path.join(LOCAL_IMAGE_DIR, filename)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return None

    try:
        # 打开并返回图片
        img = Image.open(file_path)
        return img
    except Exception:
        # 图片格式错误或无法打开
        return None


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

    支持的占位符格式：
        - [image,url=...]  网络图片，从 URL 下载
        - [image,file=...] 本地图片，从固定目录读取

    参数:
        text: 原始文本字符串
        images: 当前图像列表（会被修改，仅成功加载的图像会追加）

    返回:
        一个片段列表，每个片段是一个字典，类型为 "text" 或 "image"。
    """
    # 匹配两种格式，捕获类型和值
    pattern = re.compile(r'\[image,(?P<type>url|file)=(?P<value>[^\]]+)\]')
    segments = []
    last_end = 0

    for match in pattern.finditer(text):
        # 添加匹配之前的文本片段
        start, end = match.span()
        if start > last_end:
            segments.append({"type": "text", "text": text[last_end:start]})

        # 处理占位符
        img_type = match.group("type")
        value = match.group("value")
        img = None

        if img_type == "url":
            img = open_image_from_url(value)
        else:  # file
            img = open_image_from_local(value)

        if img is not None:
            images.append(img)
            segments.append({"type": "image", "image": img})
        else:
            segments.append({"type": "text", "text": "[发送了一张图片]"})

        last_end = end

    # 添加剩余的文本
    if last_end < len(text):
        segments.append({"type": "text", "text": text[last_end:]})

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
                         "[image,url=https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=EhR_ayMnShcGVQPOQTMKHfh54aNkPxjavysg_woo6K6y8--_kwMyBHByb2RQgL2jAVoQf6Ti2Q5jv-irQV0FhNBBLHoCxoCCAQJneg&spec=0&rkey=CAQSMIYIOjzrxb3eTCVG5osnrvoCVRkVzu0Kfso8iV7HfsZBtCTWi9LdV0dRUCu6EDOMOw]"
                         "[image,file=Arisu_00.png]"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is another image: [image,file=saiba-midori-saiba-momoi.jpg]"}
            ]
        }
    ]

    initial_images = []
    new_messages, new_images = process_messages(test_messages, initial_images)

    import pprint

    pprint.pprint(new_messages)
    print(f"Total images successfully loaded: {len(new_images)}")
