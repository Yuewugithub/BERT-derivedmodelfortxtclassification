import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix
from transformers import DistilBertTokenizer
from utils import get_parser
from data import convert_to_inputdata, read_data, trans_inputdata
from Distilbertmodel import DistilBert4TC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def predict_on_batch(text_list, args):

    tokenizer = DistilBertTokenizer.from_pretrained(args.bert_model)

    InputData_list = convert_to_inputdata(tokenizer,  text_list_x = text_list, label_list = None, mode="single")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:",device)

    predict_model_file = os.path.join(args.predict_dir, "pytorch_model.bin")

    model_state_dict = torch.load(predict_model_file)
    model = DistilBert4TC.from_pretrained(args.bert_model, num_labels = args.num_labels, state_dict = model_state_dict)
    model.to(device)

    all_input_ids, all_input_masks, all_segment_ids = trans_inputdata(InputData_list, mode = 'test')

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long).to(device)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long).to(device)

    eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
                                 batch_size=args.predict_batch_size)

    model.eval()

    target_list = []
    score_list = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc = 'TC Evaluating'):

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            batch_logits = model(input_ids=input_ids, attention_mask=input_mask)
            logits = F.softmax(batch_logits.detach().cpu(), dim=1)
            print(logits)
            result = torch.max(logits, dim=1, keepdims=False)
            pred_target = result.indices.tolist()
            pred_score = result.values.tolist()

            target_list += pred_target
            score_list += pred_score

    return target_list

def eval():

    # 处理数据
    args = get_parser()
    text_list, label_list = read_data(args.eval_file)
    target_list = predict_on_batch(text_list,args)

    CM = confusion_matrix(label_list, target_list)

    print("Confusion Matrix:\n", CM)

    correct = 0

    for i in range(len(CM)):
        correct += CM[i][i]

    acc = correct / len(label_list)
    print("Accuracy:", acc)

if __name__ == "__main__":

    args = get_parser()
    text_list = ['The New Fodcha DDoS Botnet Adds Extortion Feature to its Arsenal.',"10387061-1.v1 XMRig Cryptocurrency Mining Software.","Stuxnet_ Zero victims The identity of the companies targeted by the first known cyber-weapon."]
    target_list = predict_on_batch(text_list, args)
    print(target_list)
    # [1, 0]