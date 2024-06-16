
import torch
from dataloader_mimiciv import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from sklearn import metrics
from transformers import T5Tokenizer,AutoTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup
from model_muta_DD import MUTA
import torch
import json
import copy
import time
import argparse
import socket
import datetime
import pandas as pd
from accelerate import Accelerator
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
start = time.time()
SEED = 3407 #gpu23 model 2
torch.manual_seed(SEED)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="recon",help='ICD|RISK|Mortality')
parser.add_argument("--batch", default=12, type = int)
parser.add_argument("--eval", action= "store_true")
parser.add_argument("--sv_weight", action= "store_true")
parser.add_argument("--logs", action= "store_true")
args = parser.parse_args()

num_epochs = 100
max_length = 1600
pretrained = True
BATCH_SIZE = args.batch
SV_WEIGHTS = args.sv_weight
logs = args.logs
evaluation = args.eval
Freeze_t5coder = True
Freeze_TST = False
patch_len = 8
num_patch = 125
stride = 8
num_query_tokens = 8
trunction_len = num_query_tokens
# CUDA_VISIBLE_DEVICES="1" accelerate launch trainer_mimiciv/trainer_promptmedts_stg2.py --task recon --batch 12

accelerator = Accelerator(
    mixed_precision="no",
    device_placement=True,
)

date = str(datetime.date.today())
save_dir= "xxx"
sv_model_id = "muta_stg2"
base_model_id = "emilyalsentzer/Bio_ClinicalBERT" 
# base_model_id = "bert-base-uncased"

decoder_model_id = "google/flan-t5-small"
sv_decoder_model_id = decoder_model_id.split("/")[-1]
sv_base_model_id = base_model_id.split("/")[-1]
gpuid = socket.gethostname()
save_name = f"{sv_model_id}_{sv_decoder_model_id}_{date}_basemodel-{sv_base_model_id}_{args.task}_trunction-{trunction_len}_{SEED}_{gpuid}_task-{args.task}"
log_file_name = f'xx.txt'


start_epoch = 0
print("#####################")
print(f"Server: {gpuid}, TASK: {args.task}, date: {date}, base_model_id: {base_model_id}, decoder_model_id: {decoder_model_id}, \
                Freeze_t5coder: {Freeze_t5coder},   SEED: {SEED}, model_name: {sv_model_id}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE} "+'\n')
print("#####################")

Best_F1 = 0.6

t5_weight_dir = "xxx"

promptmedts_stg1_weight_dir = "xxx"
ts_weight_dir = "xxx"


device = accelerator.device


start_epoch = 0

if evaluation:
    pretrained = False
    SV_WEIGHTS = False
    Logging = False
    weight_dir = "/home/comp/cssniu/promptt5/promptmedts/weights/mimiciv/promptmedts_stg2_flan-t5-small_2024-05-17_basemodel-Bio_ClinicalBERT_recon_trunction-24_3407_gpu21_task-recon_epoch_4_loss_0.0606_f1_micro_0.6894_f1_macro_0.655.pth"
else:
    with open(f'{log_file_name}', 'w') as log_file:
        log_file.write(f"Server: {gpuid}, TASK: {args.task}, date: {date}, base_model_id: {base_model_id}, decoder_model_id: {decoder_model_id}, \
                    Freeze_t5coder: {Freeze_t5coder}, Freeze_TST: {Freeze_TST}   SEED: {SEED}, model_name: {sv_model_id}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE} "+'\n')
        log_file.close()
tokenizer_t5 = T5Tokenizer.from_pretrained(decoder_model_id)
tokenizer_bert = AutoTokenizer.from_pretrained(base_model_id,do_lower_case=True, local_files_only=True)


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [sq[0].shape[0] for sq in data]
    numeric_data = [i[0].tolist() for i in data]
    numeric_data = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in numeric_data],batch_first = True, padding_value=0)
    disease_names = [f"### Diagnosed result: {i[1]}" for i in data]
    # text = [i[1] for i in data]
    text = [f"Diagnose disease from the following medical notes: \n{d[2]}\n" for d in data]
    labels = [i[3] for i in data]
    lab_description = [i[4] for i in data]
    return numeric_data,disease_names,text,lab_description,labels

def fit(epoch,model,dataloader,optimizer,scheduler,flag='train'):
    global Best_F1,Best_Roc,patch_len,num_patch,stride
   
    if flag == 'train':
        model.train()


    else:
        model.eval()

    batch_loss_list = []
    y_list = []
    pred_list_f1 = []
    model = model.to(device)
   

    for i,(numeric_data,disease_names,text,lab_description,labels) in enumerate(tqdm(dataloader)):
        # if i == 30: break
        label = torch.tensor(np.array(labels)).to(torch.float32).squeeze().to(device)
        if flag == "train":
            with torch.set_grad_enabled(True):
                numeric_data = torch.tensor(numeric_data).to(torch.float32).to(device)
                numeric_data = numeric_data.view(numeric_data.shape[0],num_patch,numeric_data.shape[-1],patch_len)
                text_input_t5 = tokenizer_t5(text, truncation = True, return_tensors="pt",pad_to_max_length=True, padding="max_length",max_length = max_length).to(device)
                label_input = tokenizer_t5(disease_names, return_tensors="pt",padding=True).to(device)
                loss,mm_input,lab_feats,lab_prompts,text_feats = model(numeric_data,label_input,text_input_t5)
                accelerator.backward(loss)
                scheduler.step()
                # loss.backward(retain_graph = True)
                optimizer.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                numeric_data = torch.tensor(numeric_data).to(torch.float32).to(device)
                numeric_data = numeric_data.view(numeric_data.shape[0],num_patch,numeric_data.shape[-1],patch_len)
                text_input_t5 = tokenizer_t5(text, truncation = True, return_tensors="pt",pad_to_max_length=True, padding="max_length",max_length = max_length).to(device)
                label_input = tokenizer_t5(disease_names, return_tensors="pt",padding=True).to(device)
                loss,mm_input,lab_feats,lab_prompts,text_feats = model(numeric_data,label_input,text_input_t5)
                # loss = torch.zeros(1).to(device)
                # text_embed = model.t5_decoder.encoder.embed_tokens(text_input_t5["input_ids"])
                output_sequences = model.t5_decoder.generate(
                    inputs_embeds = mm_input,
                    # input_ids = text_input_t5["input_ids"],
                    num_beams = 2,
                    max_length = 500,
                    temperature = 0.7,
                    num_return_sequences = 1,
                    # do_sample=False,
                    # length_penalty=-1,
                )  
                pred_labels = tokenizer_t5.batch_decode(output_sequences, skip_special_tokens=True)
                # print(disease_names)
                # print(pred_labels)
                # print("..........................")

                pred = []
                for pred_label in pred_labels:
                    s_pred = [0]*len(target_diagnosis_name_list)
                    for i,d in enumerate(target_diagnosis_name_list):  
                        if d in pred_label:
                            s_pred[i] = 1  
                    pred.append(s_pred) 

                pred = np.array(pred)   
                y = np.array(label.cpu().data.tolist())
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  

    if flag == "test" or flag == "dev":

        y_list = np.vstack(y_list)
        pred_list_f1 = np.vstack(pred_list_f1)
        acc = metrics.accuracy_score(y_list,pred_list_f1)
        precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
        recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
        precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
        recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

        f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
        f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
        end = time.time()
        running_time = end - start
        print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} | ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss))
    if flag == "dev":
        if logs:
            with open(f'{log_file_name}', 'a+') as log_file:
                log_file.write("PHASE: {} EPOCH : {} | Running time: {} |  Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, running_time, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss)+'\n')
                log_file.close()
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_micro_{round(float(f1_micro),4)}_f1_macro_{round(float(f1_macro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)

    
if __name__ == '__main__':

    train_dataset = PatientDataset('mimicdata/mimic4/', flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    dev_dataset =  PatientDataset('mimicdata/mimic4/', flag="dev")
    devloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    target_diagnosis_name_list = train_dataset.risk_list
    print(train_dataset.__len__())
    print(dev_dataset.__len__())

    model = MUTA(trunction_len, Freeze_t5coder = Freeze_t5coder, Freeze_TST = Freeze_TST, num_query_tokens = num_query_tokens,decoder_model_id = decoder_model_id,base_model_id = base_model_id)  # doctest: +IGNORE_RESULT
    
    if pretrained:
        t5_wights = {}
        for key, param in torch.load(t5_weight_dir,map_location=torch.device(device)).items():
            if "t5_decoder" in key:
                t5_wights[key.replace("t5_decoder.","")] = param

        model.load_state_dict(torch.load(promptmedts_stg1_weight_dir,map_location=torch.device(device)), strict=False)
        model.load_state_dict(torch.load(ts_weight_dir,map_location=torch.device(device)), strict=False)
        model.t5_decoder.load_state_dict(t5_wights, strict=True)

    optimizer = AdamW(model.parameters(True), lr=2e-5, eps = 1e-8, weight_decay = 0.05)

    len_dataset = train_dataset.__len__()
    total_steps = (len_dataset // BATCH_SIZE) * 100 if len_dataset % BATCH_SIZE == 0 else (len_dataset // BATCH_SIZE + 1) * num_epochs 
    warm_up_ratio = 0.05 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 200, num_training_steps = total_steps)
    model, optimizer,scheduler, train_dataloader,devloader = accelerator.prepare(model, optimizer,scheduler, trainloader,devloader)
    print("model param numbers: ",sum(p.numel() for p in model.parameters()))


    if evaluation:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device)), strict=False)
        print("loading weight: ",weight_dir)
        
        fit(1,model,devloader,optimizer,scheduler,flag='dev')
     
    else:
        for epoch in range(start_epoch,num_epochs):
            fit(epoch,model,trainloader,optimizer,scheduler,flag='train')
            fit(epoch,model,devloader,optimizer,scheduler,flag='dev')

            