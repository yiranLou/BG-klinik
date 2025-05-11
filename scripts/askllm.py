#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from openai import OpenAI
from openai import OpenAIError

# 可以根据模型上下文长度自行调整（字符数）
MAX_CHUNK_SIZE = 32000  # 更大的上下文窗口，适应现代模型能力

SYSTEM_PROMPT = (
    "你是一个贴心且专业的助手，只输出对用户有用的最终答案，不要输出任何内部的思考过程。"
)


def load_material(path: str) -> str:
    """读取材料文件并返回文本内容。"""
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, max_chars: int = MAX_CHUNK_SIZE) -> list[str]:
    """将长文本分段，仅当超过上下文窗口大小时才分段。"""
    # 如果文本总长度小于上下文限制，直接返回整个文本
    if len(text) <= max_chars:
        print(f"文本长度（{len(text)} 字符）小于上下文限制（{max_chars} 字符），不进行分段")
        return [text]
    
    print(f"文本长度（{len(text)} 字符）超过上下文限制（{max_chars} 字符），将进行分段")
    
    # 按段落分割，尽量保持段落完整性
    result = []
    current_chunk = ""
    
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        para = para.strip() + "\n\n"
        
        # 如果当前段落本身超过上下文长度，需要按字符强制分割
        if len(para) > max_chars:
            # 如果当前chunk不为空，先保存
            if current_chunk:
                result.append(current_chunk.strip())
                current_chunk = ""
            
            # 长段落按字符切分
            for i in range(0, len(para), max_chars):
                result.append(para[i:i+max_chars].strip())
        
        # 如果添加这个段落会超过限制，先保存当前chunk，然后开始新的chunk
        elif len(current_chunk) + len(para) > max_chars:
            result.append(current_chunk.strip())
            current_chunk = para
        
        # 否则直接添加到当前chunk
        else:
            current_chunk += para
    
    # 不要忘记最后一个chunk
    if current_chunk:
        result.append(current_chunk.strip())
    
    print(f"文本已被分成 {len(result)} 段")
    for i, chunk in enumerate(result):
        print(f"第 {i+1} 段长度: {len(chunk)} 字符")
    
    return result


def main():
    # 直接在代码中设置问题和材料路径
    default_question = "这是 .osim文件，先告诉我通常结构是怎么样的，然后下面是我文件的内容，解释一下"  # 修改为您想要的问题
    default_material_path = r"C:\Users\polo_\Desktop\llm_test.txt"  # 修改为您的材料文件路径
    
    parser = argparse.ArgumentParser(
        description="通用的 OpenAI 聊天脚本，支持长材料自动拆分"
    )
    parser.add_argument(
        "-q", "--question",
        help="要问模型的内容。如果不指定，会使用代码中设置的默认问题。"
    )
    parser.add_argument(
        "-m", "--material-file",
        help="可选，包含背景材料的本地文本文件路径"
    )
    args = parser.parse_args()

    # 获取 API Key 和创建客户端
    api_key = "sk-c4065faadcca4ccbac4cd00fe3ebd7df"
    if not api_key:
        print("API Key 不能为空。")
        return
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 获取用户提问
    question = args.question or default_question
    if not question:
        print("提问不能为空，脚本退出。")
        return

    # 可选读取材料
    material_text = ""
    material_path = args.material_file or default_material_path
    if material_path:
        try:
            material_text = load_material(material_path)
            print(f"材料加载成功，总长度: {len(material_text)} 字符")
        except Exception as e:
            print(f"读取材料文件失败：{e}")
            return

    responses = []
    if material_text:
        chunks = chunk_text(material_text)
        
        for idx, chunk in enumerate(chunks, start=1):
            print(f"[{idx}/{len(chunks)}] 正在处理第 {idx} 段材料…")
            user_content = f"材料：\n{chunk}\n\n用户提问：{question}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ]
            try:
                resp = client.chat.completions.create(
                    model="qwen-plus-latest",
                    messages=messages,
                )
                print(f"第 {idx} 段处理完成")
            except OpenAIError as e:
                print(f"调用模型出错：{e}")
                return
            responses.append(resp.choices[0].message.content)
    else:
        user_content = f"用户提问：{question}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]
        try:
            resp = client.chat.completions.create(
                model="qwen-plus",
                messages=messages,
            )
        except OpenAIError as e:
            print(f"调用模型出错：{e}")
            return
        responses.append(resp.choices[0].message.content)

    # 打印最终结果
    print("\n=== 模型回复 ===\n")
    print("\n\n".join(responses))


if __name__ == "__main__":
    main()
