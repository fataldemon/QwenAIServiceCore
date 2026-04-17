import base64
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


def open_image_from_base64(base64_str: str) -> Optional[Image.Image]:
    """从 base64 字符串加载图像，支持 data URL 前缀"""
    if base64_str.startswith("data:image"):
        try:
            base64_str = base64_str.split(",", 1)[1]
        except IndexError:
            return None
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return img
    except Exception:
        return None


def process_text(text: str, images: List[Image.Image]) -> str:
    """
    处理一个文本字符串，将其中的图像占位符替换为 "<|vision_start|><|image_pad|><|vision_end|>" 文本。
    成功加载的图片会追加到 images 列表中。

    处理文本中的图像占位符，支持三种格式：
        [image,url=...]
        [image,file=...]
        [image,base64=...]

    参数:
        text: 原始文本字符串
        images: 当前图像列表（会被修改，仅成功加载的图像会追加）

    返回:
        替换后的文本字符串
    """
    pattern = re.compile(r'\[image,(?P<type>url|file|base64)=(?P<value>[^\]]+)\]')

    def replacer(match):
        img_type = match.group("type")
        value = match.group("value")
        img = None

        if img_type == "url":
            img = open_image_from_url(value)
        elif img_type == "file":
            img = open_image_from_local(value)
        else:  # base64
            img = open_image_from_base64(value)

        if img is not None:
            images.append(img)
            return "<|vision_start|><|image_pad|><|vision_end|>"
        else:
            return "[发送了一张图片]"

    return pattern.sub(replacer, text)


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
                  {"role": str, "content": str}
        images:   可选的已有图像列表，默认为空列表。成功加载的图像将追加到此列表之后。

    返回:
        (new_messages, new_images):
            new_messages: 处理后的消息列表，content 中的文本项中的占位符已被替换为 "<|vision_start|><|image_pad|><|vision_end|>" 或失败文本。
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

        original_content = msg.get("content")
        new_content = process_text(original_content, new_images)

        new_msg["content"] = new_content
        new_messages.append(new_msg)

    return new_messages, new_images


# 使用示例
if __name__ == "__main__":
    test_messages = [
        {
            "role": "user",
            "content": "（老师对爱丽丝说）我发了几张图片"
                    "[image,url=https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=EhR_ayMnShcGVQPOQTMKHfh54aNkPxjavysg_woo6K6y8--_kwMyBHByb2RQgL2jAVoQf6Ti2Q5jv-irQV0FhNBBLHoCxoCCAQJneg&spec=0&rkey=CAQSMIYIOjzrxb3eTCVG5osnrvoCVRkVzu0Kfso8iV7HfsZBtCTWi9LdV0dRUCu6EDOMOw]"
                    "[image,file=Arisu_00.png]"
        },
        {
            "role": "assistant",
            "content": "Here is another image: [image,file=saiba-midori-saiba-momoi.jpg]"
        }
    ]

    initial_images = []
    new_messages, new_images = process_messages(test_messages, initial_images)

    import pprint

    pprint.pprint(new_messages)
    print(f"Total images successfully loaded: {len(new_images)}")
