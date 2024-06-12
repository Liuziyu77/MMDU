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

import os
import json
from collections import defaultdict

def calculate_scores(folder_path):
    # 初始化一个字典来存储总分和计数
    total_scores = defaultdict(float)
    total_count = 0

    # 初始化一个字典来存储每个文件的结果
    file_scores = {}

    # # 遍历文件夹中的所有文件
    # for filename in os.listdir(folder_path):
    #     # 只处理 .json 文件
    #     if filename.endswith('.json'):
    #         file_path = os.path.join(folder_path, filename)

    # 遍历文件夹及其所有子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 只处理 .json 文件
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)

            try:
                # 读取 JSON 文件
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    # 初始化一个字典来存储当前文件的总分和计数
                    file_total_scores = defaultdict(float)
                    file_count = 0
    
                    # 计算当前文件每个值的总和
                    for item in data:
                        try:
                            for k in ['Creativity', 'Richness', 'Visual Perception', 'Logical Coherence', 'Answer Accuracy', 'Image Relationship Understanding']:
                                assert k in item 
                            file_total_scores['Creativity'] += item.get('Creativity')
                            file_total_scores['Richness'] += item.get('Richness')
                            file_total_scores['Visual Perception'] += item.get('Visual Perception')
                            file_total_scores['Logical Coherence'] += item.get('Logical Coherence')
                            file_total_scores['Answer Accuracy'] += item.get('Answer Accuracy')
                            file_total_scores['Image Relationship Understanding'] += item.get('Image Relationship Understanding')
                            file_total_scores['Overall Score'] += item.get('Overall Score')
                            file_count += 1
                        except Exception as e:
                            print({e})

                    if file_count!=0:
                        # 计算当前文件每个值的均分
                        file_averages = {key: total / file_count for key, total in file_total_scores.items()}
                        file_scores[filename] = {'averages': file_averages, 'totals': dict(file_total_scores), 'count': file_count}
        
                        # 更新总分和计数
                        for key, total in file_total_scores.items():
                            total_scores[key] += total
                        total_count += file_count
            except Exception as e:
                print({e})

    # 计算所有 JSON 文件的总均分
    overall_averages = {key: total / total_count for key, total in total_scores.items()}

    # 返回结果
    return file_scores, {'averages': overall_averages, 'totals': dict(total_scores), 'count': total_count}

# 文件夹路径
folder_path = './Claude3_claude_opus_results/'

# 计算分数
file_scores, overall_scores = calculate_scores(folder_path)

# 打印每个文件的结果
for filename, scores in file_scores.items():
    print(f"File: {filename}")
    print(f"  Averages: {scores['averages']}")
    print(f"  Totals: {scores['totals']}")
    print(f"  Count: {scores['count']}")

# 打印总体结果
print("Overall:")
print(f"  Averages: {overall_scores['averages']}")
print(f"  Totals: {overall_scores['totals']}")
print(f"  Count: {overall_scores['count']}")
print('\t'.join([str(s) for s in overall_scores['totals'].values()]))
