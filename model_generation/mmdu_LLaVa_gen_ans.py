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


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


disable_torch_init()

### lora sft代码
model_path = "your_model_path"
model_base = "your_model_base_path"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base, model_name
)


def eval_model(query, image_files, model, model_name, history, conv_mode=None, sep = ",", temperature=0, top_p=None, num_beams=1, max_new_tokens=2048):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if '<image-placeholder>' in query:
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = "llava_v0"
    meta_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

    messages = history.copy()
    new_conversation = [
        {"from": "Human", "content": qs},
        {"from": "Assistant", "content": ""}
    ]
    messages.extend(new_conversation)
    prompt = ""
    for i, message in enumerate(messages):
        if i >= len(messages) - 2:  # 检查是否是最后一个message
            if message["from"] == "Human":
                prompt += "###Human:"
            elif message["from"] == "Assistant":
                prompt += "###Assistant:"
            prompt += message["content"]
            prompt += "\n"
        else:
            prompt += message["content"]
            prompt += "\n"
    prompt = meta_prompt + prompt
    # print(prompt)
            
    # conv = conv_templates[conv_mode].copy()
    # print(conv)
    # conv.append_message(conv.roles[0], qs)
    # print(conv)
    # conv.append_message(conv.roles[1], None)
    # print(conv)
    # prompt = conv.get_prompt()
    # print(prompt)

    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    ### 四张图堆叠为 [4, 3, 336, 336]
    ### test = image_processor(images, return_tensors='pt')['pixel_values']
    ### print(test.shape)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    # print(images_tensor.shape)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    new_conversation = [
        {"from": "Human", "content": qs},
        {"from": "Assistant", "content": outputs}
    ]
    history.extend(new_conversation)

    return outputs, history



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
        q = q.replace("<ImageHere>", "<image-placeholder>")
        q = q.lstrip()
        print(RED+q+RESET)
        print(images)
        with torch.cuda.amp.autocast():
            response, history = eval_model(query=q, image_files=images, model=model, model_name=model_name, history=history)
        print(GREEN+response+RESET)

        record_data["conversations"][index*2+1]["value"] = response
    
    data_id = item["id"] 
    file_path = f"{model_answer_save_path}/{data_id}.json"
    with open(file_path, "w") as json_file:
        json.dump(record_data, json_file)