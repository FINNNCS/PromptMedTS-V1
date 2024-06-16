
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_patchtst import PatchTSTEncoder
from transformers import AutoTokenizer, AutoModel,BertLMHeadModel,T5ForConditionalGeneration,T5Config,AutoModelForSeq2SeqLM
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
import torch.distributed as dist

import math


class PENCBASE(Blip2Base):
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, weight_name, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(weight_name)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        prompt_encoder = BertLMHeadModel.from_pretrained(
           weight_name, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.weight =  nn.parameter.Parameter(prompt_encoder.bert.embeddings.word_embeddings.weight[:num_query_token,:].clone().detach())
        encoder_config = BertConfig.from_pretrained(weight_name)
        return prompt_encoder, query_tokens,


class MUTA(PENCBASE):
	def __init__(self,trunction_len,Freeze_promptts = False, Freeze_t5coder = True, Freeze_TST = True, prompt_encoder_hidden_size = 768, enc_dim = 768, num_features = 11, max_seq_len = 1000,patch_len = 8, num_patch = 125, num_query_tokens = 32,decoder_model_id = "google/flan-t5-small",base_model_id = "emilyalsentzer/Bio_ClinicalBERT"):
		super(MUTA, self).__init__()
		self.num_query_tokens = num_query_tokens
		self.trunction_len = trunction_len
		self.ts_encoder = PatchTSTEncoder(num_features,prompt_encoder_hidden_size,num_patch,patch_len)
		if Freeze_TST:
			for name, param in self.ts_encoder.named_parameters():
				param.requires_grad = False
			# self.ts_encoder = self.ts_encoder.eval()
			# self.ts_encoder.train = self.disabled_train
			print("freeze TST")

		self.prompt_encoder, self.lab_tokens = self.init_Qformer(num_query_tokens, prompt_encoder_hidden_size, base_model_id, 2)
		tokenizer = AutoTokenizer.from_pretrained(base_model_id,do_lower_case=True, local_files_only=True)

		self.prompt_encoder.resize_token_embeddings(len(tokenizer))
		state_dict = self.prompt_encoder.state_dict()

		for name, param in self.prompt_encoder.named_parameters():
			if "_query" in name:
					key_orig = name.replace("_query", "")
					param.data.copy_(state_dict[key_orig])

		self.lab_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)
		self.text_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)
		self.labd_proj = nn.Linear(self.prompt_encoder.config.hidden_size, enc_dim)
		self.ts_decoder = nn.Linear(prompt_encoder_hidden_size, max_seq_len)
		self.temp = nn.Parameter(0.07 * torch.ones([]))
		self.itm_head = nn.Linear(self.prompt_encoder.config.hidden_size, 2)

		self.t5_decoder =  AutoModelForSeq2SeqLM.from_pretrained(decoder_model_id)
	
		self.prompt_proj = nn.Linear(prompt_encoder_hidden_size, self.t5_decoder.config.hidden_size)

		if Freeze_promptts:
			for name, param in self.prompt_encoder.named_parameters():
				param.requires_grad = False
			print("freeze promptts")

		if Freeze_t5coder:
			for name, param in self.t5_decoder.named_parameters():
				param.requires_grad = False
			self.t5_decoder = self.t5_decoder.eval()
			print("freeze T5")


	def disabled_train(self, mode=True):
		"""Overwrite model.train with this function to make sure train/eval mode
		does not change anymore."""
		return self

	def forward(self,lab_x = None,label_input = None,text_inputt5 = None):
		lab_feats = self.ts_encoder(lab_x)
		
		lab_attention_mask = torch.ones(lab_feats.size()[:-1], dtype=torch.long, device=lab_feats.device)

		lab_tokens = self.lab_tokens.expand(lab_feats.shape[0], -1, -1)


		lab_discrete_query_outputs = self.prompt_encoder.bert(
		query_embeds=lab_tokens,
		encoder_hidden_states=lab_feats,
		encoder_attention_mask=lab_attention_mask,
		return_dict=True,
		use_cache=True,
		)
		prmopt_embedding = self.prompt_proj(lab_discrete_query_outputs.last_hidden_state)[:,:self.trunction_len,:]
		prmopt_mask = torch.ones(prmopt_embedding.size()[:-1], dtype=torch.long).to(prmopt_embedding.device)

		mm_input = torch.cat((prmopt_embedding,self.t5_decoder.encoder.embed_tokens(text_inputt5["input_ids"])),axis = 1)
		mm_mask = torch.cat((prmopt_mask,text_inputt5["attention_mask"]),axis = 1)

		output = self.t5_decoder(inputs_embeds=mm_input,attention_mask = mm_mask, labels=label_input["input_ids"])
		loss_gen = output.loss

		return loss_gen,mm_input,lab_feats,prmopt_embedding,self.t5_decoder.encoder.embed_tokens(text_inputt5["input_ids"])
