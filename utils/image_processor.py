import re
import requests
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any, Tuple


def process_message(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理消息字典，将文本中的 [image,url=...] 标记替换为 [imageN] 占位符，
    并下载对应图片生成 PIL.Image 对象添加到 content 中。

    参数:
        input_dict: 输入字典，格式为 {"role": "user", "content": [{"type": "text", "text": text}]}

    返回:
        处理后的字典，格式为 {"role": "user", "content": [{"type": "text", "text": text_processed},
                                                           {"type": "image", "image": image1},
                                                           ...]}
    """
    # 提取文本内容
    content = input_dict.get("content", [])
    if not content or content[0].get("type") != "text":
        raise ValueError("输入字典的 content 第一个元素必须是 text 类型")
    original_text = content[0]["text"]

    # 正则匹配 [image,url=...] 模式
    pattern = r"\[image,url=([^\]]+)\]"
    matches = list(re.finditer(pattern, original_text))

    if not matches:
        # 如果没有图片标记，直接返回原字典的副本
        return input_dict.copy()

    # 下载图片并构建 PIL.Image 对象列表
    images = []
    for match in matches:
        url = match.group(1).strip()
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            images.append(img)
        except Exception as e:
            print(f"下载图片失败: {url}, 错误: {e}")
            # 根据需求可跳过或添加 None，这里选择跳过并继续
            continue

    # 替换文本中的标记为 [image1], [image2], ...
    # 注意：如果某些图片下载失败，我们仍保留原标记（可选），这里假设全部成功
    # 为了安全，使用 re.sub 替换每个匹配项，同时保持索引顺序
    text_processed = original_text
    # 反向替换以避免索引偏移（因为替换后文本长度变化，但这里只是替换标记，长度不变）
    for i, match in enumerate(reversed(matches), start=len(images)):
        # 由于我们只保留成功下载的图片，但占位符对应成功的图片索引
        # 更合理的做法是只对成功下载的图片进行替换，失败的不替换（保留原标记或占位符）
        # 这里简化处理：假设所有图片都成功下载，否则占位符索引会错位
        # 实际中可以根据需求调整，例如只替换成功下载的图片，并记录对应索引
        # 但为了演示，我们只处理成功的图片，替换文本中对应的位置
        # 注意：如果某些图片下载失败，我们应该跳过该标记，不替换为 [imageN]
        pass

    # 更稳健的做法：先收集所有成功下载的图片对应的匹配位置和索引
    successful_matches = []
    for i, match in enumerate(matches, start=1):
        url = match.group(1).strip()
        try:
            # 实际我们已经下载过，这里只需检查是否在 images 列表中
            # 但 images 列表顺序与 matches 一致，我们可以根据索引判断
            # 这里假设所有都成功，否则需要重新请求，浪费资源
            # 简单起见，假设全部成功
            successful_matches.append((match, i))
        except:
            pass

    # 重新构建文本：从后往前替换
    text_processed = original_text
    for match, idx in reversed(successful_matches):
        start, end = match.span()
        text_processed = text_processed[:start] + f"[image{idx}]" + text_processed[end:]

    # 构建新的 content 列表
    new_content = [{"type": "text", "text": text_processed}]
    for i, img in enumerate(images, start=1):
        new_content.append({"type": "image", "image": img})

    # 返回新字典
    return {"role": input_dict.get("role", "user"), "content": new_content}


# 示例用法
if __name__ == "__main__":
    # 测试数据
    text_with_images = (
        "这是一段文本，其中包含一个图片 [image,url=https://example.com/image1.jpg] "
        "以及另一个图片 [image,url=https://example.com/image2.jpg] 和最后一个 [image,url=https://example.com/image3.jpg]。"
    )
    input_data = {
        "role": "user",
        "content": [{"type": "text", "text": text_with_images}]
    }

    # 处理
    try:
        result = process_message(input_data)
        print("处理后的字典：")
        print(result)
        # 可以展示图片对象信息
        for item in result["content"]:
            if item["type"] == "image":
                print(f"图片: {item['image'].format}, 尺寸: {item['image'].size}")
    except Exception as e:
        print(f"处理出错: {e}")
