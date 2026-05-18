import fcntl
import re

from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
import pickle
from template import _get_args

args = _get_args()
model = SentenceTransformer(args.embedding_path, device='cuda')

# character表示对应角色 subject表示主题：["setting", "expression", "behaviour", "memory"]
DOC_FOLDER = """embedding/{character}/{subject}/"""
VECTOR_FOLDER = """embedding/{character}/{subject}/vector/"""
IDENTITY_FILE = """embedding/{character}/identity.mem"""


def remove_reference_url(text: str) -> str:
    """
    移除字符串末尾的 <reference_url:...> 标签。

    参数:
        text: 原始字符串，末尾可能包含 <reference_url:https://..., http://..., ...>

    返回:
        去除标签后的字符串，并去除尾部多余空白。
    """
    # 匹配从 <reference_url: 开始到第一个 > 结束的内容
    pattern = r'<reference_url:[^>]*>'
    # 替换为空字符串，并去除尾部空白（如换行、空格）
    cleaned = re.sub(pattern, '', text).strip()
    return cleaned


def read_as_content(file_name: str, doc_folder: str) -> str:
    with open(doc_folder + file_name, 'r', encoding="utf-8") as file:
        file_content = file.read()
    return file_content


def write_as_memory(file_name: str, doc_folder: str, content: str):
    with open(doc_folder + file_name, 'a', encoding="utf-8") as file:
        if not content.endswith('\n'):
            content += '\n'
        file.write(content)


# 将faiss的索引index写到文件中，返回文件名
def write_index(index, vector_folder: str) -> str:
    faiss.write_index(index, vector_folder + 'index.faiss')
    return vector_folder + 'index.faiss'


def generate_vector(character: str, subject: str) -> str:
    if subject not in ["setting", "expression", "behaviour", "memory", "knowledge"]:
        return "subject incorrect"

    doc_folder = DOC_FOLDER.format(character=character, subject=subject)
    vector_folder = VECTOR_FOLDER.format(character=character, subject=subject)

    # 确保向量目录存在（即使已存在也继续重建，覆盖旧数据）
    os.makedirs(vector_folder, exist_ok=True)

    # 收集所有 .mem 文件内容
    content_parts = []
    if os.path.exists(doc_folder):
        for file_name in os.listdir(doc_folder):
            if file_name.endswith(".mem"):
                content_parts.append(read_as_content(file_name, doc_folder))
    content = "\n".join(content_parts)

    # 如果没有有效内容，清空旧向量文件后返回
    if not content.strip():
        for f in ["materials.pkl", "tags_map.pkl", "paragraphs.pkl", "srch_embeddings.npy", "index.faiss"]:
            fp = os.path.join(vector_folder, f)
            if os.path.exists(fp):
                os.remove(fp)
        return "empty"

    # 分割段落并过滤空行
    raw_paragraphs = [p for p in content.split("\n") if p.strip()]
    paragraphs: List[str] = []
    tags: List[str] = []
    tags_map: dict = {}

    for para in raw_paragraphs:
        if "##" in para:
            parts = para.split("##")
            clean_para = parts[0].strip()
            # 提取标签，过滤空标签，并去重（段落内相同标签只处理一次）
            tag_list = [t.strip() for t in parts[1:] if t.strip()]
            unique_tags = set(tag_list)
            current_idx = len(paragraphs)  # 当前段落即将存放的索引
            for tag in unique_tags:
                if tag not in tags:
                    tags.append(tag)
                    tags_map[tag] = [current_idx]
                else:
                    # 避免重复添加相同段落索引
                    if current_idx not in tags_map[tag]:
                        tags_map[tag].append(current_idx)
            paragraphs.append(clean_para)
        else:
            paragraphs.append(para)

    # 构建搜索材料：所有段落 + 所有标签（标签会作为独立检索项，映射到对应的段落）
    search_materials = paragraphs + tags

    # 保存元数据文件
    with open(os.path.join(vector_folder, 'tags_map.pkl'), 'wb') as f:
        pickle.dump(tags_map, f)
    with open(os.path.join(vector_folder, 'materials.pkl'), 'wb') as f:
        pickle.dump(search_materials, f)
    with open(os.path.join(vector_folder, 'paragraphs.pkl'), 'wb') as f:
        pickle.dump(paragraphs, f)

    # 生成向量并构建 FAISS 索引
    try:
        search_embeddings = model.encode(search_materials)
    except Exception as e:
        print(f"向量编码失败: {e}")
        return "error"

    np.save(os.path.join(vector_folder, "srch_embeddings.npy"), search_embeddings)
    dimension = search_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(search_embeddings)
    write_index(index, vector_folder)  # 该函数应通过 faiss.write_index 保存为 index.faiss

    return "success"


def add_knowledge(content: str, character: str) -> str:
    content = remove_reference_url(content)
    doc_folder = DOC_FOLDER.format(character=character, subject="knowledge")
    vector_folder = VECTOR_FOLDER.format(character=character, subject="knowledge")

    # 保存原始内容到 .mem 文件（保持原有逻辑）
    write_as_memory(file_name="knowledge.mem", doc_folder=doc_folder, content=content)

    # 使用文件锁防止并发写入
    lock_path = os.path.join(vector_folder, ".add_lock")
    os.makedirs(vector_folder, exist_ok=True)
    with open(lock_path, "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        return _add_knowledge_locked(content, character, vector_folder)


def _add_knowledge_locked(content: str, character: str, vector_folder: str) -> str:
    # 检查所有必要文件是否存在
    required_files = ["materials.pkl", "tags_map.pkl", "paragraphs.pkl", "index.faiss"]
    if not all(os.path.exists(os.path.join(vector_folder, f)) for f in required_files):
        # 缺少文件，全量重建
        return generate_vector(character, "knowledge")

    # 加载旧数据
    with open(os.path.join(vector_folder, 'materials.pkl'), 'rb') as f:
        materials = pickle.load(f)
    with open(os.path.join(vector_folder, 'tags_map.pkl'), 'rb') as f:
        tags_map = pickle.load(f)
    with open(os.path.join(vector_folder, 'paragraphs.pkl'), 'rb') as f:
        paragraphs_old = pickle.load(f)

    # 加载旧索引，用于后续增量添加
    index = faiss.read_index(os.path.join(vector_folder, 'index.faiss'))

    # 一致性检查
    if len(materials) != index.ntotal:
        # 数据不一致，全量重建
        return generate_vector(character, "knowledge")

    # 处理新内容：分割段落，过滤空行
    raw_paragraphs = [p for p in content.split("\n") if p.strip()]
    materials_num = len(materials)
    new_paragraphs = []
    new_tags = []
    tags_map = tags_map.copy()  # 避免修改原对象直到确认成功

    for i, para in enumerate(raw_paragraphs):
        if "##" in para:
            parts = para.split("##")
            clean_para = parts[0].strip()
            tag_list = [t.strip() for t in parts[1:] if t.strip()]
            # 标签去重（段落级别）
            unique_tags = set(tag_list)
            for tag in unique_tags:
                if tag not in new_tags:
                    new_tags.append(tag)
                # 避免重复添加相同段落索引
                idx = materials_num + len(new_paragraphs)  # 当前段落即将添加的位置
                if idx not in tags_map.setdefault(tag, []):
                    tags_map[tag].append(idx)
            new_paragraphs.append(clean_para)
        else:
            new_paragraphs.append(para)

    # 如果没有新增任何有效内容，直接返回
    if not new_paragraphs and not new_tags:
        return "success"

    # 生成新向量（段落 + 标签）
    to_encode = new_paragraphs + new_tags
    new_embeddings = model.encode(to_encode)

    # 增量添加到 FAISS 索引
    index.add(new_embeddings)

    # 更新 materials 和 paragraphs_old
    materials.extend(new_paragraphs + new_tags)
    paragraphs_old.extend(new_paragraphs)

    # 写回文件
    with open(os.path.join(vector_folder, 'materials.pkl'), 'wb') as f:
        pickle.dump(materials, f)
    with open(os.path.join(vector_folder, 'tags_map.pkl'), 'wb') as f:
        pickle.dump(tags_map, f)
    with open(os.path.join(vector_folder, 'paragraphs.pkl'), 'wb') as f:
        pickle.dump(paragraphs_old, f)

    # 保存索引
    faiss.write_index(index, os.path.join(vector_folder, 'index.faiss'))

    # 可选：如果需要保留原始 embedding 文件（供 debug），可以增量保存，但非必须
    # 这里省略 srch_embeddings.npy 的维护，因为 FAISS 索引已足够

    return "success"


def get_identity(character: str) -> list:
    identity_file = IDENTITY_FILE.format(character=character)
    with open(identity_file, 'r', encoding="utf-8") as f:
        file_content = f.read()
    identities = file_content.split("\n")
    print(identities)
    return identities


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def vector_search(question: str, top_k: int, character: str, subject: str, instruct: str) -> tuple[list[str], list[int]]:
    task = instruct
    question = get_detailed_instruct(task, question)
    vector_folder = VECTOR_FOLDER.format(character=character, subject=subject)
    if not (os.path.exists(vector_folder + "materials.pkl") and os.path.exists(vector_folder + "tags_map.pkl")):
        return [""], [0]
    if subject == "setting":
        for identity in get_identity(character):
            question = question.replace(identity, "你")
    with open(vector_folder + 'materials.pkl', 'rb') as f:
        materials = pickle.load(f)
    with open(vector_folder + 'tags_map.pkl', 'rb') as f:
        tags_map = pickle.load(f)
    index = faiss.read_index(vector_folder + 'index.faiss')
    search = model.encode([question])
    # 防止请求的 top_k*3+1 超出索引中的向量总数
    n_total = index.ntotal
    search_k = min(top_k * 3 + 1, n_total)
    accuracy, matches = index.search(search, search_k)
    print(accuracy, " ", matches)
    result = []
    result_index_list = []
    log_info = ""
    for i in matches[0]:
        # 检查索引 i 是否在 materials 范围内
        if i < 0 or i >= len(materials):
            print(f"警告: 检索返回的索引 {i} 超出 materials 范围 (0~{len(materials)-1})，已跳过")
            continue
        answer = materials[i].strip()
        if tags_map.get(answer) is None:
            if i not in result_index_list:
                result_index_list.append(int(i))
            log_info += f'编号{i};'
        else:
            for j in tags_map.get(answer):
                # 检查映射出的 j 是否在 materials 范围内
                if j < 0 or j >= len(materials):
                    print(f"警告: tags_map 中的索引 {j} 超出 materials 范围，已跳过")
                    continue
                log_info += f'定位到tag：{answer},编号{j};'
                if j not in result_index_list:
                    result_index_list.append(int(j))

    # 过滤掉所有可能越界的索引（其实上面已过滤，但再保一遍）
    valid_indices = [idx for idx in result_index_list if 0 <= idx < len(materials)]

    # 如果有效索引不足 top_k，则用第一个有效索引重复填充；若无任何有效索引，用 0 占位
    if len(valid_indices) < top_k:
        print(f"警告: 有效索引只有 {len(valid_indices)} 个，不足 top_k={top_k}，将使用第一个有效索引填充剩余位置")
        if valid_indices:
            pad_index = valid_indices[0]
        else:
            pad_index = 0
            # 确保 pad_index 不会越界（如果 materials 为空则特殊处理）
            if len(materials) == 0:
                print("错误: materials 为空，无法返回任何结果")
                return [""] * top_k, [0] * top_k
        while len(valid_indices) < top_k:
            valid_indices.append(pad_index)

    # 取前 top_k 个索引
    topk_indices = valid_indices[:top_k]
    result = [materials[idx].strip() for idx in topk_indices]

    print(f"IndexList={topk_indices[::-1]}.  {log_info}  ")
    # 返回队首 top_k 个元素，倒序输出（最后一个是相关度最高的）
    return result, topk_indices[::-1]


def find_material_by_index(index_list: list[int], character: str, subject: str) -> list[str]:
    vector_folder = VECTOR_FOLDER.format(character=character, subject=subject)
    with open(vector_folder + 'materials.pkl', 'rb') as f:
        materials = pickle.load(f)
    result = []
    for index in index_list:
        result.append(materials[index].strip())
    return result


def reorganize_index(base_list: list[int], append_list: list[int], max_length: int) -> list[int]:
    for member in append_list:
        if member in base_list:
            base_list.remove(member)
            base_list.append(member)
        else:
            base_list.append(member)
    return base_list[-max_length:]


def process_embedding(content: str, top_k: int, character: str,
                      client_buffer: list[int], max_length: int, client_information: str = "",) -> tuple:
    search_result, server_embedding_index_list = vector_search(
        question=content,
        character=character,
        subject="setting",
        top_k=top_k,
        instruct='给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息'
    )
    embedding_index = reorganize_index(
        base_list=client_buffer,
        append_list=server_embedding_index_list,
        max_length=max_length
    )
    server_embedding_list = find_material_by_index(
        index_list=embedding_index,
        character=character,
        subject="setting"
    )
    knowledge, knowledge_index_list = vector_search(
        question=content,
        character=character,
        subject="knowledge",
        top_k=3,
        instruct='给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息')
    full_knowledge = f"这些是你知道的事实：{server_embedding_list}\n这些是你了解的知识：{knowledge}\n{client_information}\n"
    return full_knowledge, embedding_index


def check_emotion(emotion: str, character: str) -> str:
    emotion_text = emotion.replace("【", "").replace("】", "")
    vector_folder = VECTOR_FOLDER.format(character=character, subject="expression")
    with open(vector_folder + 'materials.pkl', 'rb') as f:
        materials = pickle.load(f)
    if emotion_text in materials:
        print(f"===验证表情：{emotion_text}--->表情有效===")
        return emotion
    else:
        index = faiss.read_index(vector_folder + 'index.faiss')
        search = model.encode([get_detailed_instruct('找到与给出的表情表达情感最相近的表情', emotion_text)])
        # search = model.encode([emotion_text])
        accuracy, matches = index.search(search, 1)
        i = matches[0][0]
        final_emotion = materials[i].strip()
        print(f"===验证表情：{emotion_text}--->表情替换为：{final_emotion}===")
        return f"【{final_emotion}】"


if __name__ == "__main__":
    # print("Setting:" + generate_vector("tendou_arisu", "setting"))
    # print("Expression:" + generate_vector("tendou_arisu", "expression"))
    # print("Behavior:" + generate_vector("tendou_arisu", "behaviour"))
    # print("Memory:" + generate_vector("tendou_arisu", "memory"))
    print("Knowledge:" + generate_vector("tendou_arisu", "knowledge"))
    print(vector_search("温泉乡", 3, "tendou_arisu", "setting",
                        "给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息"))
    # print(vector_search("", 3, "tendou_arisu", "knowledge",
    #                     "给一句对话内容，找到涉及对话中出现的话题、人物、地点、组织、学校等信息的相关信息"))
    # print(check_emotion("关心", "tendou_arisu"))



