# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:46:07 2023

@author: 10638
"""

from mhc_dataload import AAdataset
import math
from tqdm import tqdm
import argparse
from model import RLpMIEC, RLpMIECConfig
import pandas as pd
import torch
import numpy as np
from numpy import random
import os
import sys
import pickle
import pdb
from RLtrainer import TrainerConfig,Trainer
from att_novdw import mydataset,myconfig,att_model
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--prior_name', type=str, default = 'prior_model', help="The prior model", required=False)
        parser.add_argument('--miec_name', type=str, default = 'MIEC_generator', help="MIEC generator", required=False)
        parser.add_argument('--score_name', type=str, default = 'Score_model', help="The scoring function", required=False)
        parser.add_argument('--batch_size', type=int, default = 64, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 26, help="number of layers", required=False) 
        parser.add_argument('--block_size', type=int, default = 10, help="number of layers", required=False)   
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--tmp', type=float, default = 1, help="temperature", required=False)
        parser.add_argument('--max_epochs', type=int, default=10,help="total epochs", required=False)
        parser.add_argument('--is_pretrain', type=bool, default=False, #args bug(bool('False')=True,被认为是str))给予任意值都是True，有需求再使用
                            help="properties to be used for condition", required=False)
        parser.add_argument('--save_name', type=str, default='trained_model',
                            help="modelname", required=False)
        parser.add_argument('--sigma', type=int, default = 1, help="sigma", required=False)
        parser.add_argument('--learning_rate', type=float,
                            default=6e-3, help="learning rate", required=False)
        args = parser.parse_args()
        set_seed(42)
        AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
        # itos = { i:ch for i,ch in enumerate(chars) }
        stoi = { ch:i for i,ch in enumerate(AA_set) }
        itos = { i:ch for i,ch in enumerate(AA_set) }
        with open('./dataset/require_seq_180.pkl','rb') as f:
            seqlist=pickle.load(f)
        seqlistAA=[]
        for seq in seqlist:
            seqlistAA.append([stoi[i] for i in seq])
        # prob=stat_prob(data.iloc[:,0],stoi)
        max_seq_len=10
        max_len=12
        num_miec=180

        mconf = RLpMIECConfig(args.vocab_size, args.block_size,num_miec=num_miec,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,seqlist=seqlistAA)
        miec_conf = RLpMIECConfig(args.vocab_size, args.block_size,num_miec=num_miec,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,seqlist=seqlistAA,is_mc=True)
        miec_model = RLpMIEC(miec_conf)
        miec_model.load_state_dict(torch.load('./model/'+args.miec_name+'.pt'))
        miec_model.to('cuda:0')
        prior = RLpMIEC(mconf)
        prior.load_state_dict(torch.load('./model/'+args.prior_name+'.pt'))
        prior.to('cuda:0')
        agent = RLpMIEC(mconf)
        agent.load_state_dict(torch.load('./model/'+args.prior_name+'.pt'))
        agent.to('cuda:0')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
        myconf = myconfig()
        score_model = att_model(myconf)
        score_model.load_state_dict(torch.load(f'./model/{args.score_name}.pt'))
        score_model.eval()
        score_model.to('cuda:0')

        tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,tmp = args.tmp,
                                num_workers=8, ckpt_path=f'./model/{args.save_name}.pt', block_size=max_len, generate=False,sigma=args.sigma)
        RLtrainer = Trainer(prior,agent,score_model,
                            tconf,stoi,itos,miec_model)
        RLtrainer.train()

        
        
        
        
        
        
        
        
        
        
        
        
        