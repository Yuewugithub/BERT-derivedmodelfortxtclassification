from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载预训练模型的tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(r'E:/transformers/dataaugBERT')
model = BertForMaskedLM.from_pretrained(r'E:/transformers/dataaugBERT')

# 创建一个包含两个[MASK]的句子
input_text = "i [MASK] [MASK] go for a walk in the park ."

def predict(input_text):
# 使用tokenizer将句子转换为模型的输入数据
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    print(tokenizer.convert_tokens_to_ids(input_text))
    print(input_ids)

    # 用模型进行预测
    output = model(input_ids)

    # 获取每个位置最高概率的token
    predicted_indices = torch.argmax(output.logits, dim=2)

    # 将预测的token ID转换为词汇
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices[0].tolist())

    print(predicted_tokens)