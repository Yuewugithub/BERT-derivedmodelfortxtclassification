import random

# 打开原始文本文件
with open('data/train.txt', 'r', encoding="utf-8") as file:
    original_text = file.readlines()

# 打乱句子顺序
random.shuffle(original_text)

# 创建一个新的文本文件来保存打乱顺序后的文本
with open('datashuffled.txt', 'w',encoding="utf-8") as file:
    file.writelines(original_text)

print("句子顺序已被打乱，并保存在文档：shuffled.txt")