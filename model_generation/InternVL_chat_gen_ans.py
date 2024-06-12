import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
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

from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def init_model():
    path = "/mnt/hwfile/mllm/zangyuhang/hf_models/InternVL-Chat-V1-5"
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model.tokenizer = tokenizer
    return model

def get_response_concat(model, question, image_path_list, history, max_new_tokens=2048, max_num=5):
    generation_config = dict(
        num_beams=1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    pixel_values_list = [load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda() for image_path in image_path_list]
    pixel_values = torch.cat(pixel_values_list, dim=0)
    if len(history) == 0:
        response, history = model.chat(model.tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    else:
        response, history = model.chat(model.tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
    return response, history


model = init_model()


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
                images = img_paths
            else:
                images = img_paths
            q = q.replace("Image1: <ImageHere>.", "")
            q = q.replace("Image2: <ImageHere>.", "")
            q = q.replace("Image3: <ImageHere>.", "")
            q = q.replace("Image4: <ImageHere>.", "")
            q = q.replace("Image5: <ImageHere>.", "")
            q = q.lstrip()
            print(RED+q+RESET)
            print(images)
            # with torch.cuda.amp.autocast():
            response, history = get_response_concat(model, question = q, image_path_list = images, history = history)
            print(GREEN+response+RESET)

            record_data["conversations"][index*2+1]["value"] = response
        
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        with open(file_path, "w") as json_file:
            json.dump(record_data, json_file)
    except Exception as e:
        print({e})