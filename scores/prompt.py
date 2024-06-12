import io
import re
import os
import json
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image
import difflib
import time
from collections import defaultdict, Counter
import matplotlib.image as mpimg

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色

from openai import OpenAI
import base64
api_key = "your OpenAI API key"
def mllm_openai(query, images):
    # client = OpenAI(api_key=api_key, base_url=api_base)
    client = OpenAI(api_key=api_key)

    # 读取图像文件并将其转换为 base64 编码的字符串列表
    base64_images = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            base64_images.append(base64_image)

    response = client.chat.completions.create(
        model = "gpt-4o,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images],
                ],
            },
        ],
        max_tokens=4096,
    )

    assistant_response = response.choices[0].message.content

    return assistant_response

meta_prompt = """
You are an assistant skilled at evaluating the quality of creative text.
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. You'll need to assess the response on the following dimensions: Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy and Image Relationship Understanding. We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below:

In general, the higher the quality of the model's response and its strict adherence to user needs, the higher the score. Responses that do not meet user needs will receive lower scores.

Scoring rules:
Creativity:
Scores 1-2 when there is no innovation or uniqueness in the content.
Scores 3-4 when providing partially original content but with low creative quality.
Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
Scores 7-8 when having novelty and high-quality content.
Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.

Richness:
Scores 1-2 when lacking depth and breadth, with very limited information.
Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
Scores 5-6 when limited in depth and breadth but provides basic necessary information.
Scores 7-8 when providing depth and useful additional information.
Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.

Visual Perception:
Scores 1-2 when the description of the visual information in the image contains errors or is significantly inconsistent with the content of the image.
Scores 3-4 When the description of the visual information in the image reflects only a small amount of the image's information and contains some errors.
Scores 5-6 when the description of the visual information in the image includes the basic information of the image but contains minimal information.
Scores 7-8 when the description of the visual information in the image matches the image well and is rich in content, providing a substantial amount of information about the image.
Scores 9-10 when the description of the visual information in the image not only matches the image but also is more detailed and informative compared to the reference answer, providing more information about the image.

Logical Coherence:
Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
Scores 7-8 when excellent logical handling, very few errors.
Scores 9-10 when flawless logic, impeccable in handling complexity, and significantly higher logical coherence compared to the reference answer.

Answer Accuracy
Scores 1-2 when the answer is significantly inconsistent with the question or contains obvious errors.
Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question.
Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, exceeding expectations.

Image Relationship Understanding:
Scores 1-2 when there are significant errors or confusion in distinguishing and describing different images, unable to correctly identify and relate the content of the images.
Scores 3-4 when the description of different images reflects only minimal distinguishing information, contains some errors and confusion, and fails to clearly differentiate and relate the images.
Scores 5-6 when the description of different images includes basic distinguishing information, is able to correctly identify and relate the images in a basic manner, but the information provided is minimal and lacks detail.
Scores 7-8 when the description of different images is accurate and detailed, clearly distinguishing and relating the images, with rich content that points out the main commonalities and differences between the images.
Scores 9-10 when the description of different images is not only accurate and detailed but also provides richer information and analysis, clearly distinguishing and relating the images, more comprehensively pointing out the commonalities and differences between the images compared to the reference answer.

Overall Score:
Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
Scores 7-8 when performing well in all dimensions.
Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.

Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, add the score for that dimension. Finally, at the end of your response, in the format of the dictionary (including brackets), return all your scoring results, ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
"""

question_begin_prompt = "[Question]"
reference_begin_prompt = "[The Start of Reference Answer]"
reference_end_prompt = "[The End of Reference Answer]"
answers_begin_prompt = "[The Start of Assistant’s Answer]"
answers_end_prompt = "[The End of Assistant’s Answer]"

with open('path/to/benchmark.json', 'r', encoding='utf-8') as f:
    benchmarks = json.load(f)
print(len(benchmarks))

### local_index from 0-10. Divide the 110 conversations into 10 parts for testing.
local_index = 0
### scores and reasons save path
file_save_fold = "./Claude3_gpt4_turbo_results"
### model generation answer
file_get_fold = "./Claude3_results"
local_range = str(local_index*10)+"-"+str((local_index+1)*10)
benchmarks = benchmarks[local_index*10: (local_index+1)*10]
print(len(benchmarks))
benchmarks_ids = [item["id"] for item in benchmarks]
print(benchmarks_ids)

scores = 0
questions_num = 0
for benchmark_i in tqdm(range(len(benchmarks))):
    try:
        data_id = benchmarks[benchmark_i]["id"]
        file_path = f"{file_save_fold}/{local_range}/{data_id}.json"
        if os.path.exists(file_path):
            print(f"File exists: {file_path}, skipping this iteration.")
            continue  # 跳过该次循环

        ### 读取 benchamrk 题目和 AI 生成的对话
        conv_benchmarks = benchmarks[benchmark_i]["conversations"]
        id = benchmarks[benchmark_i]["id"]
        with open(f'{file_get_fold}/{id}.json', 'r', encoding='utf-8') as f:
            Model_answers = json.load(f)
        conv_ai_assistant = Model_answers["conversations"]
        images = benchmarks[benchmark_i]["image"]
        # plot_images(images)
        print(images)
        
        questions = []
        reference = []
        ai_assistant = []
        for i in conv_benchmarks:
            if i["from"] == "user":
                questions.append(i["value"])
            if i["from"] == "assistant":
                reference.append(i["value"])
        for i in conv_ai_assistant:
            if i["from"] == "assistant":
                ### 正常模型的评估代码
                ai_assistant.append(i["value"])
                ### Qwen的评估代码
                # ai_assistant.append(i["value"][0]["text"])

        all_result_dict = []
        for j in range(len(questions)):
            try:
                promt = meta_prompt + question_begin_prompt + '\n' + questions[j] + '\n\n' + reference_begin_prompt + '\n' + reference[j] + '\n' + reference_end_prompt + '\n\n' + answers_begin_prompt + '\n' + ai_assistant[j] + '\n' + answers_end_prompt
                print(RED+questions[j]+RESET)
                print("Length of Ground Truth:"+ str(len(reference[j])))
                print("Length of Reference:"+ str(len(ai_assistant[j])))
                print("Benchmarks:"+str(benchmark_i)+" Question:"+str(j))
                print(reference[j])
                
                responese = mllm_openai(promt, images)
                start_index = responese.find('{')
                end_index = responese.rfind('}') + 1
                dictionary_str = responese[start_index:end_index]
                result_dict = eval(dictionary_str)
                print(responese)
                # print(GREEN+str(result_dict)+RESET)
                
                scores += result_dict["Overall Score"]
                questions_num += 1
                print(scores/questions_num)
                all_result_dict.append(result_dict)
            except Exception as e:
                print({e})
        file_path = f"{file_save_fold}/{local_range}/{id}.json"
        with open(file_path, "w") as json_file:
            json.dump(all_result_dict, json_file)
    except Exception as e:
        print({e})