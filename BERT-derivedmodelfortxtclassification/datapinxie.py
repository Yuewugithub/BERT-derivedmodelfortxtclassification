import random

# 函数：插入随机字符
def insert_random_char(word):
    position = random.randint(0, len(word))
    char = chr(random.randint(97, 122))  # 生成随机小写字母
    return word[:position] + char + word[position:]

# 函数：删除随机字符
def delete_random_char(word):
    if len(word) > 0:
        position = random.randint(0, len(word) - 1)
        return word[:position] + word[position + 1:]
    else:
        return word

# 函数：替换随机字符
def replace_random_char(word):
    if len(word) > 0:
        position = random.randint(0, len(word) - 1)
        char = chr(random.randint(97, 122))
        return word[:position] + char + word[position + 1:]
    else:
        return word

# 函数：交换相邻字符
def swap_adjacent_chars(word):
    if len(word) > 1:
        position = random.randint(0, len(word) - 2)
        return word[:position] + word[position + 1] + word[position] + word[position + 2:]
    else:
        return word

# 打开原始文本文件
with open('data/train.txt', 'r', encoding="utf-8") as file:
    original_text = file.readlines()

# 创建一个新的文本文件来保存带有拼写错误的文本
with open('modifiedpinxie.txt', 'w',encoding="utf-8") as file:
    for line in original_text:
        words = line.split()
        modified_words = []
        for word in words:
            # 随机选择一个错误类型
            error_type = random.choice(["insert", "delete", "replace", "swap"])

            if error_type == "insert":
                new_word = insert_random_char(word)
            elif error_type == "delete":
                new_word = delete_random_char(word)
            elif error_type == "replace":
                new_word = replace_random_char(word)
            else:
                new_word = swap_adjacent_chars(word)

            modified_words.append(new_word)

        modified_line = ' '.join(modified_words)
        file.write(modified_line + '\n')

print("拼写错误已添加到文档：modified.txt")