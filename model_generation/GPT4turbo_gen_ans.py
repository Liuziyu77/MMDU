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


from openai import OpenAI
import base64
api_key = "your OpenAI API"
def mllm_openai(query, images, conversation_history):
    client = OpenAI(api_key=api_key)

    base64_images = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            base64_images.append(base64_image)

    # conversation_history.append({"role": "user", "content": query, "images_count": len(images)})

    # 创建一个modified_conversation_history的副本,用于传递给API,只取role和content字段
    modified_conversation_history = [
        {"role": message["role"], "content": message["content"]}
        for message in conversation_history
    ]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(modified_conversation_history)

    if len(images)!=0:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images],
            ],
        })
    
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images],
            ],
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        conversation_history.append({"role": "user", "content": [{"type": "text", "text": query}]})
        
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=4096,
    )

    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, conversation_history


with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

model_answer_save_path = "path/to/save_answer"

benchmarks = [item for item in benchmarks.values()]

for item in tqdm(benchmarks):
    record_data = item.copy()
    ### 获取图像路径
    img_paths = item["image"]
    # print(img_paths)
    # plot_images(img_paths)

    ### 获取问题
    conv = item["conversations"]
    questions = []
    for i in conv:
        if i["from"] == "user":
            questions.append(i["value"])
    # print(questions)

    try:
        pics_number = 0
        history = []
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
            
            # print(RED+q+RESET)
            print(tag_number)
            print(images)
            with torch.cuda.amp.autocast():
                response, history = mllm_openai(query=q, images=images, conversation_history=history)
            # print(GREEN+response+RESET)
    
            record_data["conversations"][index*2+1]["value"] = response
        # print(record_data)
        
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        with open(file_path, "w") as json_file:
            json.dump(record_data, json_file)
    except Exception as e:
        print({e})