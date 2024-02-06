

import torch
from torch.utils.data import Dataset

import numpy as np
import math 

torch.manual_seed(42)
def randomize_aaseq(aaseq):
    AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
    aa_len=len(aaseq)
    random_seq=''
    for i in range(aa_len):
        random_seq+=np.random.choice(AA_set)
    return random_seq
        
class AAdataset(Dataset):
    def __init__(self,args,data,chars,block_size,aug_prob = 0.5,seq_idx=1,cluster_idx=0,miec_idx=3):
        vocab_size=len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data

        self.aug_prob = aug_prob  
        self.seq=np.array(data.iloc[:,seq_idx]).copy()
        if miec_idx !=None:                         
            self.miec1 = torch.tensor(np.array(data.iloc[:,miec_idx::4]),dtype=torch.float32).unsqueeze(1)
            self.miec2 = torch.tensor(np.array(data.iloc[:,miec_idx+1::4]),dtype=torch.float32).unsqueeze(1)
            self.miec3 = torch.tensor(np.array(data.iloc[:,miec_idx+2::4]),dtype=torch.float32).unsqueeze(1)
            self.miec4 = torch.tensor(np.array(data.iloc[:,miec_idx+3::4]),dtype=torch.float32).unsqueeze(1)
            self.miec  = torch.concat([self.miec1,self.miec2,self.miec3,self.miec4],1)
        else:
            self.miec=np.zeros((len(data),4,180))
        self.cluster=np.array(data.iloc[:,cluster_idx]).copy()
    def __len__(self):
            return len(self.data)  
    def __getitem__(self, idx):
        seq,miec = self.seq[idx], self.miec[idx]   
        cluster = torch.tensor([int(self.cluster[idx])+self.vocab_size]) 
        valid_len=torch.tensor(len(seq))
        p = np.random.uniform()
        if p < self.aug_prob:
            seq=randomize_aaseq(seq)
        if len(seq)==11:
            seq='!'+seq
        else:
            seq = '!'+seq+'&'
        seq += str('<')*(self.max_len+2 - len(seq))
        dix =  [self.stoi[s] for s in seq]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        y = torch.cat([cluster,y])
        return x, y, miec,valid_len,cluster
    def __sample__(self, batch):
        n_sample = np.random.choice(range(0,len(self.seq)),batch)