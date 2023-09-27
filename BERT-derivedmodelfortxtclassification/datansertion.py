from nltk.corpus import wordnet
import random
import nltk
nltk.download("omw-1.4")
# 获取同义词
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# 打开原始文本文件
with open('data/train.txt', 'r', encoding="utf-8") as file:
    original_text = file.read()

# 将原始文本按空格拆分为单词列表
words = original_text.split()

# 替换同义词
modified_words = []
for word in words:
    synonyms = get_synonyms(word)
    if synonyms:
        modified_words.append(random.choice(synonyms))
    else:
        modified_words.append(word)

# 将替换后的单词列表重新组合为文本
modified_text = ' '.join(modified_words)

# 创建一个新的文本文件来保存替换后的文本
with open('modifiedinsersion.txt', 'w',encoding="utf-8") as file:
    file.write(modified_text)

print("同义词替换已完成，并保存在文档：modified.txt")