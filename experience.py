
import numpy as np
import torch
import pandas as pd
class Experience(object):
    def __init__(self, max_size=100,reverse=False):
        #memory ->(cluster,seq,score,nllloss,miec)
        self.memory = []
        self.max_size = max_size
        # self.voc = voc
        self.reverse =reverse
    def add_experience(self, experience):
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, seq = [], []
            for i, exp in enumerate(self.memory):
                if exp[0]+exp[1] not in seq:
                    idxs.append(i)
                    seq.append(exp[0]+exp[1])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[-3],reverse=self.reverse)
            self.memory = self.memory[:self.max_size]
    def __len__(self):
        return len(self.memory)
        
    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[2] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=(1-np.array(scores)) / (len(self.memory)-np.sum(scores)))
            sample = [self.memory[i] for i in sample]
            cluster = [x[0] for x in sample]
            seq = [x[1] for x in sample]
            scores = [x[-3] for x in sample]
            prior_likelihood = [x[-2] for x in sample]
            miec = [x[-1] for x in sample]
        return torch.tensor(seq).reshape(n,-1),torch.tensor(cluster).reshape(n,-1), torch.tensor(scores).reshape(n,-1), torch.tensor(prior_likelihood).reshape(n,-1),torch.tensor(miec).reshape(n,-1)
