import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
import io
import torch
from transformers import AutoModel, AutoTokenizer
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


def init_model():
    model_path = '/mnt/hwfile/mllm/zangyuhang/hf_models/idefics2-8b'
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path).to("cuda")
    model.processor = processor
    return model

def keep_last_n_images_and_others(input_str, n):
    # 将字符串按 <image> 分割
    parts = input_str.split("<image>")
    
    # 计算需要保留的<image>部分
    if len(parts) > n:
        kept_images = parts[-n:]
        remaining_parts = parts[:-n]
        # 重新组合保留的部分，加上 <image>
        result_str = "<image>".join(remaining_parts).replace("<image>", "") + "<image>" + "<image>".join(kept_images)
    else:
        result_str = input_str
    
    return result_str

def process_string(input_str, insertion_list, n):
    # 分割字符串为行
    lines = input_str.split("\n")
    
    # 保留最后的n个<image>并且移除其他行
    image_count = 0
    filtered_lines = []
    for line in reversed(lines):
        if "<image>" in line:
            if image_count < n:
                filtered_lines.insert(0, line)
                image_count += 1
        else:
            filtered_lines.insert(0, line)
    
    # 合并保留的行
    processed_str = "\n".join(filtered_lines)
    
    # # 保留最后一个<end_of_utterance>
    # parts = processed_str.split("<end_of_utterance>")
    # if len(parts) > 1:
    #     processed_str = "".join(parts[:-1]) + "<end_of_utterance>" + parts[-1]
    
    # 遍历字符串中的Assistant: 字符并插入列表中的字符串
    assistant_split = processed_str.split("Assistant:")
    for i in range(1, len(assistant_split)):
        if i-1 < len(insertion_list):
            assistant_split[i] = f"Assistant:{insertion_list[i-1]}{assistant_split[i]}"
        else:
            assistant_split[i] = f"Assistant:{assistant_split[i]}"
    
    # 组合处理后的字符串
    final_str = "".join(assistant_split)
    return final_str

# # 示例输入
# input_str = """User:<image><image>他们分别是什么颜色的?<end_of_utterance>
# Assistant:<end_of_utterance>
# User:在图一和图二中你看见了什么?<end_of_utterance>
# Assistant: <end_of_utterance>
# User:<image><image>他们分别是什么颜色的?<end_of_utterance>
# Assistant:"""
# insertion_list = ["这是插入的第一句话", "这是插入的第二句话"]
# n = 2

# # 调用函数并打印结果
# processed_str = process_string(input_str, insertion_list, 2)
# print(processed_str)
# processed_str = keep_last_n_images_and_others(processed_str,2)
# print(processed_str)

def get_response_concat(model, question, image_path_list, history, max_new_tokens=2048):
    # print(history)
    content = [{"type": "image"}] * len(image_path_list)
    content.append({"type": "text", "text": question},)

    messages = history.copy()
    # print(messages)
    new_input = [
        {
            "role": "user",
            "content": content,
        }
    ]
    messages.extend(new_input)
    # print(messages)
    history_answers = []
    for item in messages:
        if item["role"]=="Assistant":
            history_answers.append(item["content"])
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    # print(prompt)
    prompt = process_string(prompt, history_answers, len(image_path_list))
    prompt = keep_last_n_images_and_others(prompt, len(image_path_list))
    # print(prompt)
    inputs = model.processor(text=prompt, images=[image_path_list], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = model.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    except Exception as e:
        print(e)
        response = "Failed"

    new_messages = [
        {
            "role": "user",
            "content": content,
        },
        {
            "role": "Assistant",
            "content": response,            
        }
    ]
    history.extend(new_messages)
    # print(history)
    return response, history


model = init_model()


with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

model_answer_save_path = "path/to/save_answer"

benchmarks = [item for item in benchmarks.values()]


for item in tqdm(benchmarks):
    record_data = item.copy()
    img_paths = item["image"]

    try:
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        if os.path.exists(file_path):
            print(f"File exists: {file_path}, skipping this iteration.")
            continue  # 跳过该次循环

        ### 获取问题
        conv = item["conversations"]
        questions = []
        for i in conv:
            if i["from"] == "user":
                questions.append(i["value"])

        ### 遍历每一个问题
        pics_number = 0
        history = []
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                pics_number += tag_number
                images = img_paths[:pics_number]
            else:
                images = img_paths[:pics_number]

            q = q.replace("<ImageHere>","")
            print(RED+q+RESET)
            print(images)
            with torch.cuda.amp.autocast():
                response, history = get_response_concat(model, q, images, history = history)
            print(GREEN+response+RESET)

            record_data["conversations"][index*2+1]["value"] = response
        
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        with open(file_path, "w") as json_file:
            json.dump(record_data, json_file)
    except Exception as e:
        print({e})