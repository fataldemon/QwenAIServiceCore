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
    cleaned = re.sub(pattern, '', text).rstrip()
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


def generate_vector(character: str, subject: str):
    if subject not in ["setting", "expression", "behaviour", "memory", "knowledge"]:
        return "subject incorrect"
    content = ""
    doc_folder = DOC_FOLDER.format(character=character, subject=subject)
    vector_folder = VECTOR_FOLDER.format(character=character, subject=subject)
    if not os.path.exists(vector_folder):
        # 如果不存在，创建目录
        os.mkdir(vector_folder)
        return "empty"
    file_list = os.listdir(doc_folder)
    for file_name in file_list:
        # 读取所有文件内容为字符串
        if file_name.endswith(".mem"):
            content += read_as_content(file_name, doc_folder)
    if content == "":
        return "empty"
    # 按换行符分割为段落
    paragraphs = content.split("\n")

    # sentences = []
    tags = []
    tags_map = {}
    for i in range(len(paragraphs)):
        paragraph = paragraphs[i]
        # 对注解进行解析
        if "##" in paragraph:
            tag_list = paragraph.split("##")
            paragraphs[i] = tag_list[0]
            tag_list = tag_list[1:]
            print(tag_list)
            for tag in tag_list:
                tag = tag.strip()
                if tag not in tags:
                    tags.append(tag)
                    tags_map[tag] = [i]
                else:
                    tags_map[tag].append(i)
    print(tags_map)
    # 搜索时将注解与段落并列，制造出搜索材料，并以搜索材料为基准生成向量
    search_materials = paragraphs + tags
    if not os.path.exists(vector_folder):
        # 如果不存在，创建目录
        os.mkdir(vector_folder)
    with open(vector_folder + 'tags_map.pkl', 'wb') as f:
        pickle.dump(tags_map, f)
    with open(vector_folder + 'materials.pkl', 'wb') as f:
        pickle.dump(search_materials, f)

    with open(vector_folder + 'paragraphs.pkl', 'wb') as f:
        pickle.dump(paragraphs, f)
    # 生成向量
    search_embeddings = model.encode(search_materials)
    # 保存文件内容为向量
    np.save(vector_folder + "srch_embeddings", search_embeddings)
    dimension = search_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(search_embeddings)
    write_index(index, vector_folder)
    return "success"


def add_knowledge(content: str, character: str):
    content = remove_reference_url(content)
    doc_folder = DOC_FOLDER.format(character=character, subject="knowledge")
    vector_folder = VECTOR_FOLDER.format(character=character, subject="knowledge")
    write_as_memory(file_name="knowledge.mem", doc_folder=DOC_FOLDER.format(character=character, subject="knowledge"),
                    content=content)
    if not os.path.exists(vector_folder):
        # 如果不存在，创建目录，并全量更新向量
        os.mkdir(vector_folder)
        return generate_vector(character, "knowledge")

    # 按换行符分割为段落
    paragraphs = content.split("\n")
    tags = []
    # 读取旧数据
    with open(vector_folder + 'materials.pkl', 'rb') as f:
        materials = pickle.load(f)
    with open(vector_folder + 'tags_map.pkl', 'rb') as f:
        tags_map = pickle.load(f)
    with open(vector_folder + 'paragraphs.pkl', 'rb') as f:
        paragraphs_old = pickle.load(f)
    materials_num = len(materials)
    for i in range(len(paragraphs)):
        paragraph = paragraphs[i]
        # 对注解进行解析
        if "##" in paragraph:
            tag_list = paragraph.split("##")
            paragraphs[i] = tag_list[0]
            tag_list = tag_list[1:]
            print(tag_list)
            for tag in tag_list:
                tag = tag.strip()
                if tag not in tags:
                    tags.append(tag)
                    tags_map.setdefault(tag, []).append(materials_num + i)
                else:
                    tags_map[tag].append(materials_num + i)
    print(f"Paragraphs: {paragraphs}")
    print(f"Tags_Map: {tags_map}")

    # 搜索时将注解与段落并列，制造出搜索材料，并以搜索材料为基准生成向量
    materials += paragraphs + tags
    paragraphs_old += paragraphs
    if not os.path.exists(vector_folder):
        # 如果不存在，创建目录
        os.mkdir(vector_folder)
    with open(vector_folder + 'tags_map.pkl', 'wb') as f:
        pickle.dump(tags_map, f)
    with open(vector_folder + 'materials.pkl', 'wb') as f:
        pickle.dump(materials, f)
    with open(vector_folder + 'paragraphs.pkl', 'wb') as f:
        pickle.dump(paragraphs_old, f)
    # 读取向量
    embeddings = model.encode(paragraphs + tags)
    old_embeddings = np.load(vector_folder + "srch_embeddings.npy")
    search_embeddings = np.vstack((old_embeddings, embeddings))
    # 保存文件内容为向量
    np.save(vector_folder + "srch_embeddings", search_embeddings)
    dimension = search_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(search_embeddings)
    write_index(index, vector_folder)
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
    accuracy, matches = index.search(search, top_k*3+1)
    print(accuracy, " ", matches)
    result = []
    result_index_list = []
    log_info = ""
    for i in matches[0]:
        answer = materials[i].strip()
        if tags_map.get(answer) is None:
            if i not in result_index_list:
                result_index_list.append(int(i))
            log_info += f'编号{i};'
        else:
            for j in tags_map.get(answer):
                log_info += f'定位到tag：{answer},编号{j};'
                if j not in result_index_list:
                    result_index_list.append(int(j))
    for k in range(top_k):
        result.append(materials[result_index_list[k]].strip())
    print(f"IndexList={result_index_list[:top_k]}.  {log_info}  ")
    # print("搜索结果为：", result)  # 抛弃最后一个换行符
    return result, result_index_list[:top_k][::-1]  # 返回队首top_k个元素，倒序输出（最后一个是相关度最高的）


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



