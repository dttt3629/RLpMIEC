
 ## Highly refer from nano-Gpt,respect for Karpathy
import pandas as pd
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from model import RLpMIEC, RLpMIECConfig
from trainer import Trainer, TrainerConfig
import math
import re
import random
from mhc_dataload import AAdataset
import pickle 
import torch
import pdb
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
import pdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='mhc_data_all',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=6e-3, help="learning rate", required=False)
    parser.add_argument('--pre_save_name', type=str, default='pre_saved_model',
                        help="modelname", required=False)
    parser.add_argument('--is_pretrain', type=bool, default=False,
                        help="pretain", required=False)
    parser.add_argument('--save_name', type=str, default='trained_model',
                        help="modelname", required=False)
    parser.add_argument('--is_continue', type=bool, default=False, 
                        help="properties to be used for condition", required=False)
    parser.add_argument('--continue_save_name', type=str, default='pre_trained_model',
                        help="modelname", required=False)
    parser.add_argument('--is_mc', type=bool, default=False, 
                        help="MIEC Generator", required=False)
    args = parser.parse_args() #get metric
    set_seed(42)
    clu_size=13
    with open('./dataset/require_seq_180.pkl','rb') as f:
        seqlist=pickle.load(f)
    data = pd.read_csv('./dataset/'+args.data_name + '.csv')
    data= data.sample(frac =1).reset_index(drop=True)  #randomlised
    num_data=data.shape[0]
    train_data=data.iloc[:int(num_data*0.8),:].copy().reset_index(drop=True)
    val_data=data.iloc[int(num_data*0.8):int(num_data*0.9),:].copy().reset_index(drop=True)
    test_data=data.iloc[int(num_data*0.9):,:].copy().reset_index(drop=True)
    AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
    max_seq_len=10
    max_len=11
    num_miec=180
    stoi = { ch:i for i,ch in enumerate(AA_set) }
    seqlistAA=[]
    for seq in seqlist:
        seqlistAA.append([stoi[i] for i in seq])
    if args.is_pretrain:
        train_dataset=AAdataset(args,train_data,AA_set,max_seq_len,aug_prob =0.3,miec_idx=None)
        val_dataset=AAdataset(args,val_data,AA_set, max_seq_len,aug_prob =0,miec_idx=None)
        test_dataset=AAdataset(args,test_data,AA_set, max_seq_len,aug_prob = 0,miec_idx=None)
    elif args.is_mc:
        train_dataset=AAdataset(args,train_data,AA_set,max_seq_len,aug_prob =0)
        val_dataset=AAdataset(args,val_data,AA_set, max_seq_len,aug_prob =0)
        test_dataset=AAdataset(args,test_data,AA_set, max_seq_len,aug_prob = 0)
    else:
        train_dataset=AAdataset(args,train_data,AA_set,max_seq_len,aug_prob =0)
        val_dataset=AAdataset(args,val_data,AA_set, max_seq_len,aug_prob =0)
        test_dataset=AAdataset(args,test_data,AA_set, max_seq_len,aug_prob = 0)
    mconf = RLpMIECConfig(train_dataset.vocab_size+clu_size, max_len,num_miec=num_miec,  
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,is_pretrain=args.is_pretrain,
                        seqlist=seqlistAA,is_mc = args.is_mc)
    model = RLpMIEC(mconf)
    if  args.is_continue:
        model.load_state_dict(torch.load(f'./model/{args.pre_save_name}.pt'),strict=False)
        print('load model')
    model.to('cuda')

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_seq_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=8, ckpt_path=f'./model/{args.save_name}.pt', block_size=max_len, generate=False)
    trainer = Trainer(model, train_dataset, test_dataset,val_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos,is_pretrain=args.is_pretrain)
    df = trainer.train()
    train_data.to_csv(f"./dataset/{args.save_name}_traindata.csv",header=None,index=None)
    val_data.to_csv(f"./dataset/{args.save_name}_valdata.csv",header=None,index=None)
    test_data.to_csv(f"./dataset/{args.save_name}_testdata.csv",header=None,index=None)