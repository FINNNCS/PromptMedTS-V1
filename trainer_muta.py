import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_mm import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from transformers import AutoTokenizer
import requests
from model_muta import PromptMedTS
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import dill
import copy
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="no")

SEED = 47 

torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"

num_epochs = 50
max_length = 500
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
accumulation_step = 10
pretrained = True
Freeze = False
SV_WEIGHTS = True
evaluation = False
disease_all = True
logs = True
fp16 = False
patch_len = 8
num_patch = 125
num_query_tokens = 24
stride = 8
Best_loss = 20

loss_ration = [1,1,1]

save_dir= ""
save_name = f""
log_file_name = f'xx.txt'

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0

weight_dir = "xxx.pth"
if disease_all:
    weight_dir = "xxx.pth"


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True, local_files_only=True)
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
    weight_dir = "xx.pth"


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [sq[0].shape[0] for sq in data]
    input_x = [i[0].tolist() for i in data]
    y = [i[1] for i in data]
    input_x = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)
    text = [i[2] for i in data]
    lab_description = [i[-1] for i in data]

    return input_x,y,text,lab_description


def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]

    seq_mask = attention_mask[:,[-1]]


    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - len(input_ids[0])
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)
    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec


def fit(epoch,model,dataloader,optimizer,scheduler,flag='train'):
    global Best_loss,Best_Roc,patch_len,num_patch,stride
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()

    y_list = []
    pred_list_f1 = []
    pred_list_roc = []
    model = model.to(device)

    for i,(lab_x,labels,text_list,lab_description) in enumerate(tqdm(dataloader)):
        if flag == "train":
            with torch.set_grad_enabled(True):

                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                text_input = tokenizer(text_list, return_tensors="pt",padding=True).to(device)
                if text_input['input_ids'].shape[1] > max_length:
                        text_input = clip_text(BATCH_SIZE,max_length,text_input,device)
                elif text_input['input_ids'].shape[1] < max_length:
                        text_input = padding_text(BATCH_SIZE,max_length,text_input,device)
                labdescp_input = tokenizer(lab_description, return_tensors="pt",padding=True).to(device)

                loss_ltc,loss_ltm,loss_lm,_ = model(lab_x,text_input,labdescp_input)
                loss = loss_ration[0]*loss_ltc + loss_ration[1]*loss_ltm + loss_ration[2]*loss_lm
                loss.backward(retain_graph=True)
                if (i+1)%accumulation_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()
                batch_loss_list.append( loss.cpu().data )  
        else:
            with torch.no_grad():
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)


                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)

                text_input = tokenizer(text_list, return_tensors="pt",padding=True).to(device)
                if text_input['input_ids'].shape[1] > max_length:
                        text_input = clip_text(TEST_BATCH_SIZE,max_length,text_input,device)
                elif text_input['input_ids'].shape[1] < max_length:
                        text_input = padding_text(TEST_BATCH_SIZE,max_length,text_input,device)
                labdescp_input = tokenizer(lab_description, return_tensors="pt",padding=True).to(device)

                loss_ltc,loss_ltm,loss_lm,lm_output = model(lab_x,text_input,labdescp_input)
                loss = loss_ration[0]*loss_ltc + loss_ration[1]*loss_ltm + loss_ration[2]*loss_lm
                batch_loss_list.append( loss.cpu().data )  
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    print("PHASE: {} EPOCH : {} | Total LOSS  : {}  ".format(flag,epoch + 1,total_loss))
    if logs:
        with open(f'{log_file_name}', 'a+') as log_file:
            log_file.write("PHASE: {} EPOCH : {} | Total LOSS  : {}  ".format(flag,epoch + 1, total_loss)+'\n')
            log_file.close()
    if flag == 'test':
        if SV_WEIGHTS:
            PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(total_loss),4)}.pth"
            if total_loss < Best_loss:
                Best_loss = total_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)

if __name__ == '__main__':

    train_dataset = PatientDataset(f'xx',flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f'xx',flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    print(train_dataset.__len__())
    print(test_dataset.__len__())
    model = PromptMedTS(num_query_tokens = num_query_tokens)  # doctest: +IGNORE_RESULT
    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
        print("loading weight: ",weight_dir)

    ### freeze parameters ####
    optimizer = AdamW(model.parameters(True), lr=1e-5, eps = 1e-8, weight_decay = 0.05)

    len_dataset = train_dataset.__len__()
    total_steps = (len_dataset // BATCH_SIZE) * 100 if len_dataset % BATCH_SIZE == 0 else (len_dataset // BATCH_SIZE + 1) * num_epochs 

    warm_up_ratio = 0.1 

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    # model, optimizer,scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
    #     model, optimizer,scheduler, trainloader, testloader)
    if evaluation:
    
        fit(1,model,testloader,optimizer,scheduler,flag='test')
     
    else:
        for epoch in range(start_epoch,num_epochs):

            fit(epoch,model,trainloader,optimizer,scheduler,flag='train')
            fit(epoch,model,testloader,optimizer,scheduler,flag='test')


            