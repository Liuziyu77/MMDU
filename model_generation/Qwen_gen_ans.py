import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import io
import re
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("your_model_path", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("your_model_path", device_map="auto", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("your_model_path", trust_remote_code=True)


with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))

model_answer_save_path = "path/to/save_answer"

benchmarks = [item for item in benchmarks.values()]


for item in tqdm(benchmarks):
    record_data = item.copy()
    img_paths = item["image"]

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
            images = img_paths[pics_number : pics_number+tag_number]
            pics_number += tag_number
            for i, image in enumerate(images):
                q = q.replace('<ImageHere>', '<img>'+image+'</img>', 1) 
                
        print(RED+q+RESET)
        print(images)
        with torch.cuda.amp.autocast():
            response, history = model.chat(tokenizer, query=q, history=history)
        print(GREEN+response+RESET)

        record_data["conversations"][index*2+1]["value"] = response
    
    data_id = item["id"] 
    file_path = f"{model_answer_save_path}/{data_id}.json"
    with open(file_path, "w") as json_file:
        json.dump(record_data, json_file)