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
from transformers import AutoModelForCausalLM

import deepseek_vl
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


def init_model():
    # specify the path to the model
    model_path = "/mnt/hwfile/mllm/zangyuhang/hf_models/deepseek-vl-7b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    vl_gpt.vl_chat_processor = vl_chat_processor
    vl_gpt.tokenizer = tokenizer
    return vl_gpt


def get_response_concat(model, question, image_path_list, history, max_new_tokens=2048):
    messages = history.copy()
    # print(messages)
    conversation = [
        {
            "role": "User",
            "content": question,
            "images": image_path_list,
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]
    messages.extend(conversation)
    # print(messages)
    
    # load images and prepare for inputs
    pil_images = load_pil_images(messages)
    prepare_inputs = model.vl_chat_processor(
        conversations=messages,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=model.tokenizer.eos_token_id,
        bos_token_id=model.tokenizer.bos_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    response = model.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    new_conversation = [
        {
            "role": "User",
            "content": question,
            "images": image_path_list,
        },
        {
            "role": "Assistant",
            "content": response
        }
    ]
    history.extend(new_conversation)
    # print(history)
    return response, history


model = init_model()

with open('/path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))
print(type(benchmarks))
# print(benchmarks.keys())

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
            if tag_number ==1:
                pics_number += 1
                images = [img_paths[pics_number-1]]
            else:
                pics_number_end = pics_number+tag_number
                images = img_paths[pics_number: pics_number_end]
                pics_number += tag_number
        else:
            images = []

        q = q.replace("<ImageHere>","<image_placeholder>")
        print(RED+q+RESET)
        print(images)
        # with torch.cuda.amp.autocast():
        response, history = get_response_concat(model,question=q, image_path_list=images, history=history)
        print(GREEN+response+RESET)

        record_data["conversations"][index*2+1]["value"] = response
    
    data_id = item["id"] 
    file_path = f"{model_answer_save_path}/{data_id}.json"
    with open(file_path, "w") as json_file:
        json.dump(record_data, json_file)