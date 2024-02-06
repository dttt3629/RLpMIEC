
import math
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
import re
import pandas as pd

from torch.nn import functional as F
from utils import RL_sample
import pdb
import pickle
from experience import Experience
logger = logging.getLogger(__name__)
def try_gpu(i=0):  #@save
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1 
     # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    ckpt_path = None
    num_workers = 4 # for DataLoader
    epoch_rcd =[]
    train_l_rcd=[]
    train_la_rcd=[]
    train_lb_rcd=[]
    val_l_rcd=[]
    val_la_rcd=[]
    val_lb_rcd=[]
    avs_l_rcd=[]
    maxlist=Experience(300)
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, prior,agent, score_model,config, stoi, itos,miec_model,is_pretrain=False):
        self.score = score_model
        self.miec_model = miec_model
        self.prior = prior
        self.agent = agent
        self.config = config
        self.block_size = config.block_size
        # take over whatever gpus are on the system
        self.device = next(agent.parameters()).device
        self.stoi = stoi
        self.itos = itos
        self.is_pretrain=is_pretrain
        self.maxlist = config.maxlist
        self.batch_size=config.batch_size
        self.tmp = config.tmp
        num_gpus=4

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.agent.module if hasattr(self.agent, "module") else self.agent
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        prior, agent, config = self.prior, self.agent, self.config
        miec_model = self.miec_model
        score, maxlist = self.score, self.maxlist
        for module in prior.parameters():
            module.requires_grad = False
        raw_model = agent.module if hasattr(self.agent, "module") else agent
        optimizer = raw_model.configure_optimizers(config)
        sigma = self.config.sigma
        def run_epoch(split,epoch):
            agent.train()
            prior.eval()
            miec_model.eval()
            losses = []
            same=[]
            losses_a=[]
            losses_b=[]
            scuccess_prior=[]
            seq,miec,cluster,prior_nllloss = RL_sample(prior,miec_model, self.batch_size, self.block_size, begin=self.stoi['!'],temperature=self.tmp,sample=True)
            cluster=cluster.cpu()
            seq=seq.cpu()
            prior_nllloss=prior_nllloss.cpu()
            seqc=torch.concat([cluster,seq],1)
            seqlist=[]
            uniq=[]
            for i in range(seq.shape[0]):
                se = seq[i]
                sec = list(np.array(seqc[i])) 
                if self.stoi['!'] in se[1:] or self.stoi['<'] in se[:-2] or self.stoi['&'] in se[:-2]:
                    continue
                se2 = ''.join([self.itos[int(i)] for i in se]).replace('!','').replace('<','').replace('&','')
                if sec not in seqlist:
                        seqlist.append(sec)
                        uniq.append(i)
            miec=miec[uniq].transpose(1,2).squeeze().cpu()
            cluster=cluster[uniq].to(next(agent.parameters()).device)
            seq=seq[uniq].to(next(agent.parameters()).device)
            miec_losses = []
            device=next(agent.parameters()).device

            logits,_,_=agent(idx=seq[:,:-1],cluster=cluster)
            targets=seq[:,1:].reshape(-1,1)
            agent_nllloss=F.nll_loss(torch.log(F.softmax(logits[1],-1).reshape(-1, logits[1].size(-1))), targets.reshape(-1),reduction='none').reshape(seq.shape[0],-1).sum(-1)
            score_device=next(score.parameters()).device
            enc = seq[:,1:-1].detach().clone()
            for i in range(enc.shape[0]):
                if enc[i,-1]==1:
                    enc[i,-1]=2

            clu = cluster-23
            out,_,_ = score(idx=enc.to(score_device),prop=miec[:,1:].to(score_device),cluster=clu.squeeze().to(score_device))
            miec_score = out[:,-1,0].detach().cpu().numpy()
            miec_score=(1+miec_score/15)
            augmented_loss=prior_nllloss[uniq].to(next(agent.parameters()).device)+sigma*torch.tensor(miec_score,device=next(agent.parameters()).device)
            miec_loss = F.mse_loss(logits[0].transpose(1,2).squeeze(),miec.to(self.device).squeeze(),reduction='none').mean(dim=-1).mean(-1)
            RL_loss = torch.pow((augmented_loss-sigma/3- agent_nllloss), 2)
            a=1
            loss = RL_loss
            if len(maxlist)> 4:
                exp_seqs, exp_cluster, exp_score, exp_prior_likelihood,exp_miec = maxlist.sample(4)
                logits2,_,_=agent(idx=exp_seqs[:,:-1].to(self.device),cluster=exp_cluster.to((self.device)))
                exp_nllloss=F.nll_loss(torch.log(F.softmax(logits2[1],-1).reshape(-1, logits2[1].size(-1))), exp_seqs[:,1:].to(self.device).reshape(-1),reduction='none').reshape(exp_seqs.shape[0],-1).sum(-1)
                exp_miec_loss = F.mse_loss(logits2[0].transpose(1,2).reshape(4,-1).squeeze(),exp_miec.to(self.device).squeeze(),reduction='none').mean(dim=-1)
                exp_augmented_loss = exp_prior_likelihood.to(self.device) + sigma * exp_score.to(self.device)
                exp_RL_loss =  torch.pow((exp_augmented_loss.squeeze() -sigma/3- exp_nllloss), 2)
                exp_loss = exp_RL_loss
                agent_nllloss = torch.cat((agent_nllloss, exp_nllloss), 0)
                loss = torch.concat([loss,exp_loss],0)
            maxlist.add_experience(zip(cluster.tolist(),seq.tolist(),miec_score,prior_nllloss.tolist(),miec.tolist()))
            loss = loss.mean()
            losses.append(loss.item())
            losses_a.append(RL_loss.mean().item())
            losses_b.append(miec_loss.mean().item())
            agent.zero_grad()
            loss.backward()
            optimizer.step()
            lr = config.learning_rate 
            return loss.mean().item(),RL_loss.mean().item(),miec_loss.mean().item()
        pbar = tqdm(range(config.max_epochs))
        for epoch in tqdm(range(config.max_epochs)):
                train_loss,train_loss_a,train_loss_b = run_epoch('train',epoch)

                pbar.set_description(f"epoch {epoch+1} : train loss {train_loss},{train_loss_a},{train_loss_b}")
                
                config.train_l_rcd.append(train_loss)
                config.train_la_rcd.append(train_loss_a)
                config.train_lb_rcd.append(train_loss_b)

        if self.config.ckpt_path is not None :
            self.save_checkpoint()
        return None
