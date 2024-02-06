import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import pdb
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model,miec_model, x, steps, temperature=1.0, sample=False, top_k=None,):

    block_size = model.get_block_size()   
    model.eval()
    cluster=None
    miec=None
    for k in range(steps+1):
        if temperature<1:
            temp = float(min(0.5*(1-math.cos(math.pi*float(k/steps))+2*temperature),1.0))
        else:
            temp =1 
        if k == 0:
            logits,_,_ = model(x)
            cluster=logits[2]
            prob = F.softmax(cluster[:,0,:],dim=-1)
            cluster=torch.multinomial(prob, num_samples=1)+23
            continue
        elif k==12:

            logits2,_,_= miec_model(x[:,:-1],cluster=cluster)
            logits,_,_= model(x[:,:-1],cluster=cluster)
            miec=logits2[0]
            continue
        else:
            logits,_,_=model(x,cluster=cluster)
        logits = logits[1][:, k-1, :] / temp
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue

        x = torch.cat((x, ix), dim=1)
    return x,miec,cluster-23
@torch.no_grad()
def RL_sample(model,miec_model,batch_size, steps,begin=0, temperature=1.0, sample=False, top_k=None):
    x = torch.tensor(begin).repeat(batch_size).reshape(batch_size,1).to(next(model.parameters()).device)
    model.eval()
    cluster=None
    miec=None
    nllloss=None
    for k in range(steps+1):
        if temperature<1:
            temp = float(min(0.5*(1-math.cos(math.pi*float(k/steps))+2*temperature),1.0))
        else:
            temp =1 
        if k == 0:
            logits,_,_ = model(x)
            cluster=logits[2]
            probs = F.softmax(cluster[:,0,:],dim=-1)
            cluster=torch.multinomial(probs, num_samples=1)+23
            continue
        elif k==12:
            logits2,_,_= miec_model(x[:,:-1],cluster=cluster)
            logits,_,_= model(x[:,:-1],cluster=cluster)
            miec=logits2[0]
            continue
        else:
            logits,_,_=model(x,cluster=cluster)
        logits = logits[1][:, k-1, :] / temp
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    logits,_,_=model(x[:,:-1],cluster=cluster)
    targets=x[:,1:]
    nllloss=F.nll_loss(torch.log((F.softmax(logits[1],-1).reshape(-1, logits[1].size(-1)))), targets.reshape(-1),reduction='none').reshape(batch_size,-1).sum(-1)
    return x,miec,cluster,nllloss

def stat_prob(data,stoi):
    #start AA with prob,data is a sequence of AA
    prob=[0 for x in range(len(stoi))]
    for x in range(len(data)):
        char=data[x][0]
        prob[stoi[char]]+=1
    prob_re=[x/len(data) for x in prob]
    prob_t=torch.tensor(prob_re)
    return prob_t
