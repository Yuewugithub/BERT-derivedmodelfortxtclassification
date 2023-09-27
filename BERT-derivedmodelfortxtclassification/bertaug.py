#coding=utf-8
import torch
from transformers import BertTokenizer, BertForMaskedLM
import random
import bertprediction
def replacesentence(sentence,word_to_replace,position):
    # 定义一个句子和要替换的单词及其位置

    # 将句子分割为一个单词列表
    tokenizer = BertTokenizer.from_pretrained(r'E:/transformers/dataaugBERT')
    words = tokenizer.tokenize(sentence)

    # 确保位置在单词列表的范围内
    if 1 <= position <= len(words):
        # 将第 N 个单词替换为新的单词
        words[position - 1] = word_to_replace
    else:
        print("Position is out of range.")

    # 使用 join() 函数将单词列表重新组合成一个句子
    new_sentence = " ".join(words)

    return new_sentence

def augment_data(sentence, num_augmented_sentences,num_masks):
    datax=[]
    tokenizer = BertTokenizer.from_pretrained(r'E:/transformers/dataaugBERT')
    model = BertForMaskedLM.from_pretrained(r'E:/transformers/dataaugBERT')

    tokenized_sentence = tokenizer.tokenize(sentence)

    # 选择要替换的token
    mask_positions = random.sample(range(len(tokenized_sentence)-2), num_masks)
    print(mask_positions)
    #mask_positions=[0,1,7]
    # 进行替换
    for pos in mask_positions:
        tokenized_sentence[pos] = '[MASK]'
    masked_sentence =' '.join(tokenized_sentence)
    print(masked_sentence)
    augmented_sentences = []

    indexed_tokens = tokenizer.encode(tokenized_sentence, return_tensors='pt')

    with torch.no_grad():
        outputs = model(indexed_tokens)

        topk_values, topk_indices = torch.topk(outputs.logits, 5)

        # 初始化一个列表来存储预测的词汇
        predicted_tokens = []

        # 遍历每个位置上的预测
        for indices in topk_indices[0]:
            # 将预测的token ID转换为词汇
            tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
            predicted_tokens.append(tokens)

        #print(predicted_tokens)
        for positions in mask_positions:
            print("预测结果"+str(positions))
            for predictword in predicted_tokens[1+positions]:
                datax.append(replacesentence(sentence,predictword,positions+1))

    #     value,indice = torch.topk(outputs.logits,2,dim=2)
    #     predictions = torch.argmax(outputs.logits,dim=2)
    #     print(predictions)
    #
    #     predictions=torch.split(indice, split_size_or_sections=1, dim=2)
    #     print(predictions)
    #     for predict in predictions:
    #         print(predict.shape)
    #         predicted_tokens = tokenizer.convert_ids_to_tokens(predict.squeeze(-1)[0].tolist())
    #         print(predicted_tokens)
    #
    # return predicted_tokens
    return datax
# Example usage
with open('oringin.txt', 'r',encoding="utf-8") as file:
    original_text = file.readlines()
    for line in original_text:
        original_sentence = line
        try:
            augmented_sentences = augment_data(original_sentence, 2,3)
            for sentence in augmented_sentences:
                with open('aug.txt', 'a+',encoding="utf-8") as file1:
                    file1.write(sentence+'\n')
        except:
            continue
