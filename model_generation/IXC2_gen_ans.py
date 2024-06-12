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


from typing import List, Optional, Tuple, Union
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None
def chat(
        model,
        tokenizer,
        query: str,
        image: torch.Tensor = None,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float=1.005,
        meta_instruction:
        str = 'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.',
        **kwargs,
    ):
        if image is None:
            inputs = model.build_inputs(tokenizer, query, history, meta_instruction)
            im_mask = torch.zeros(inputs['input_ids'].shape[:2]).cuda().bool()
        else:
            if len(image)==1:
                image = model.encode_img(image[0])
                inputs, im_mask = model.interleav_wrap_chat(tokenizer, query, image, history, meta_instruction)
            else:
                encoded_images = [model.encode_img(img) for img in image]
                image = torch.cat(encoded_images, dim=0)
                inputs, im_mask = model.interleav_wrap_chat(tokenizer, query, image, history, meta_instruction)
            
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]
        outputs = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            im_mask=im_mask,
            **kwargs,
        )
        if image is None:
            outputs = outputs[0].cpu().tolist()[len(inputs['input_ids'][0]):]
        else:
            outputs = outputs[0].cpu().tolist()
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split('[UNUSED_TOKEN_145]')[0]
        history = history + [(query, response)]
        return response, history


model = AutoModel.from_pretrained('your_model_path', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('your_model_path', trust_remote_code=True)


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
    try:
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                pics_number += tag_number
                images = img_paths[:pics_number]
            else:
                images = img_paths[:pics_number]
            
            print(RED+q+RESET)
            with torch.cuda.amp.autocast():
                response, history = chat(model, tokenizer, query=q, image=images, history=history, do_sample=False)
            print(GREEN+response+RESET)
    
            record_data["conversations"][index*2+1]["value"] = response
        
        data_id = item["id"] 
        file_path = f"{model_answer_save_path}/{data_id}.json"
        with open(file_path, "w") as json_file:
            json.dump(record_data, json_file)
    except Exception as e:
        print({e})