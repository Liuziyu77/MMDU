import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

def plot_images(image_paths):
    num_images = len(image_paths)
    
    # 创建图形并显示图片
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


from openai import OpenAI
import base64
api_key = "your Qwen API key"
api_base = "your Qwen API base"


def mllm_openai(query, images, conversation_history):
    client = OpenAI(api_key=api_key, base_url=api_base)

    modified_conversation_history = [
        {"role": message["role"], "content": message["content"]}
        for message in conversation_history
    ]
    
    messages = [{"role": "system", "content": [{'type': 'text', 'text': "You are a helpful assistant."}]}]
    # messages = []
    messages.extend(modified_conversation_history)
    

    if len(images)!=0:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"{image}"}} for image in images],
            ],
        })
    
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"{image}"}} for image in images],
            ],
        })
        # print(messages)
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        conversation_history.append({"role": "user", "content": [{"type": "text", "text": query}]})

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=messages,
    )

    assistant_response = response.choices[0].message.content
    assistant_response[0]['type'] = 'text'
    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, conversation_history


with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

benchmarks = [item for item in benchmarks.values()]

model_answer_save_path = "path/to/save_answer"

import os
im2url = json.load(open('./scripts/cab357qiu.json'))
for item in tqdm(benchmarks):
    record_data = item.copy()
    img_paths = item["image"]

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
    # print(questions)

    ### 遍历每一个问题
    pics_number = 0
    history = []
    try:
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                if tag_number ==1:
                    pics_number += 1
                    images = [img_paths[pics_number-1]]
                else:
                    pics_number_end = pics_number+tag_number
                    images = img_paths[pics_number: pics_number_end]
                    pics_number += tag_number
            else:
                images = []

            print(images)
            images = [
                im2url[
                    image.split("/")[-1]
                    # image.replace("/mnt/hwfile/mllm/liuziyu/new_data/WIT/pics/", "")
                    # .replace("/mnt/hwfile/mllm/liuziyu/new_data/Dayingbaike/pics/", "")
                    ]
                for image in images
            ]
            print(RED+q+RESET)
            print(images)
            with torch.cuda.amp.autocast():
                response, history = mllm_openai(query=q, images=images, conversation_history=history)
            print(response[0]["text"])
            print(GREEN+response[0]["text"]+RESET)

            record_data["conversations"][index*2+1]["value"] = response
        
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        with open(file_path, "w") as json_file:
            json.dump(record_data, json_file)
    except Exception as e:
        print({e})