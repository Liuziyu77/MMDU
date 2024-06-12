import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import re
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


from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("your_model_path")
model = LlavaNextForConditionalGeneration.from_pretrained("your_model_path", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")


def gen_answer(query, img_list, history):

    images = [Image.open(img).resize((336,336), Image.Resampling.LANCZOS) for img in img_list]
    new_conversation = "[INST]"+ query + "[/INST]"
    if len(history)!=0:
        prompt = history[0] + new_conversation
    else:
        prompt = new_conversation

    print(prompt)
    
    inputs = processor(text=prompt, images=images, padding=True, return_tensors="pt").to(model.device)

    # Generate
    generate_ids = model.generate(**inputs, max_new_tokens=2048)
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    index = response[0].rfind('[/INST]')
    if index != -1:
        response = response[0][index + len('[/INST]'):].strip()
    else:
        response = ""

    
    if len(history)!=0:
        hisroty = [history[0] + new_conversation + response]
    else:
        hisroty = [new_conversation + response]

    return response, hisroty


with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

model_answer_save_path = "path/to/save_answer"

benchmarks = [item for item in benchmarks.values()]

import os
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
        q = q.replace("<ImageHere>", "<image>\n")
        q = q.lstrip()
        print(RED+q+RESET)
        print(images)
        with torch.cuda.amp.autocast():
            response, history = gen_answer(query=q, img_list=images, history=history)
        print(GREEN+response+RESET)

        record_data["conversations"][index*2+1]["value"] = response
    
    data_id = item["id"] 
    file_path = f"{model_answer_save_path}/{data_id}.json"
    with open(file_path, "w") as json_file:
        json.dump(record_data, json_file)