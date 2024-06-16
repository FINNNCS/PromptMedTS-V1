
import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import T5Tokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
import random
from datetime import datetime
from collections import defaultdict
import json
import inflect
import glob

import json

SEED = 2019
torch.manual_seed(SEED)

p = inflect.engine()
# train_50_bhc_with_lab.csv
class PatientDataset(object):
    def __init__(self, data_dir, Rationale_ts = False, inlcude_rationale = False,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.inlcude_rationale = inlcude_rationale
        self.numeric_dir = 'xx/mimicdata/mimic4/lab_test/'
        if flag == "train":
            self.data_dir = os.path.join(data_dir,"train_50_bhc_with_lab.csv")
        else:
            self.data_dir = os.path.join(data_dir,"dev_50_bhc_with_lab.csv")

        self.patient_df = pd.read_csv(self.data_dir)  
        self.new_text_dir = "xxx"
        print(f"Rationale dir: {self.new_text_dir }")
        self.stopword = list(pd.read_csv('xx/stopwods.csv').values.squeeze())

        self.low = [2.80000000e+01, -7.50000000e-02,  4.30000000e+01, 4.00000000e+01,
                    4.10000000e+01,  9.00000000e+01,  5.50000000e+00,  6.15000000e+01,  
                    3.50000000e+01,  3.12996266e+01, 7.14500000e+00] 
        self.up = [  92.,           0.685,         187.,         128.,   
                    113.,         106.,          33.5,        177.5,         
                    38.55555556, 127.94021917,   7.585]   
        self.interpolation = [  59.0,           0.21,         128.0,         86.0,   
            77.0,         98.0,          19.0,        118.0,         
            36.6, 81.0,   7.4]
        self.max_length = 1000
        self.feature_list = [
        'Diastolic blood pressure',
        'Fraction inspired oxygen', 
        'Glucose', 
        'Heart Rate', 
        'Mean blood pressure', 
        'Oxygen saturation', 
        'Respiratory rate',
        'Systolic blood pressure', 
        'Temperature', 
        'Weight', 
        'pH']
        self.risk_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
       

    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
   
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)
        return temp
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def __getitem__(self, idx):
        text = self.patient_df["TEXT"][idx]
        text = self.rm_stop_words(text)
        numeric_data_file = glob.glob(self.numeric_dir+str(self.patient_df["SUBJECT_ID"][idx])+"_"+ str(self.patient_df["HADM_ID"][idx])+"_episode*_timeseries.csv")[0]

        lab_dic = defaultdict(list)
        lab_description = []

        if not os.path.exists(numeric_data_file):
            numeric_data = np.array([self.interpolation]*24)
            for l in range(numeric_data.shape[-1]):
                descp = f"{self.feature_list[l]} is normal all the time"
                lab_description.append(descp.lower())

        else:
            numeric_data = pd.read_csv(numeric_data_file)[self.feature_list].values
            for l in range(numeric_data.shape[-1]):
                for s in np.array(numeric_data[:,l]):
                    if s <= self.low[l]:
                        lab_dic[l].append("low")
                    elif s > self.up[l]:
                        lab_dic[l].append("high")
                    else:
                        lab_dic[l].append("normal")
            for k in lab_dic.keys():
                risk_types = set(lab_dic[k])
                if len(risk_types) ==1 and "normal" in risk_types:
                    descp = self.feature_list[k] + f" is normal all the time"
                else:
                    if "high" in risk_types:
                        number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "high")[0]))
                        descp = self.feature_list[k] + f" is higher than normal {number} times"
                    if "low"  in risk_types:
                        number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "low")[0]))
                        descp = self.feature_list[k] + f" is lower than normal {number} times"

                    if "high" in risk_types and  "low"  in risk_types:
                        high_number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "high")[0]))
                        low_number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "low")[0]))
                        descp = self.feature_list[k] + f" is higher than normal {high_number} times and is lower than normal {low_number} times"

                lab_description.append(descp.lower())


        labels = self.patient_df[self.risk_list].values[[idx],:]
        disease_names = list(np.extract(labels == 1, self.risk_list))
        if not disease_names:
            disease_names = ["No disease is diagnosed"]
        if len(numeric_data) < self.max_length:
            numeric_data = np.concatenate((numeric_data, np.repeat(np.expand_dims(numeric_data[-1,:], axis=0),1000-len(numeric_data),axis = 0) ), axis=0)
        else:
            numeric_data = numeric_data[:self.max_length]
        lab_description = ",".join(lab_description)
        disease_names = ','.join(disease_names)

        return numeric_data,disease_names,text,labels,lab_description,self.patient_df["HADM_ID"][idx]


    def __len__(self):
        return len(self.patient_df)

