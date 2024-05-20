
import time
import pandas as pd
import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F
import pickle 
from model import RLpMIEC,RLpMIECConfig
from utils import RL_sample,sample
import math 
from tqdm import tqdm
import argparse
import pdb
from att_novdw import mydataset,myconfig,att_model
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_name', type=str, default = 'test_model', help="name of the dataset to train on", required=False)
        parser.add_argument('--prior_name', type=str, default = 'prior_model', help="using for reference to check the bias", required=False)
        parser.add_argument('--miec_name', type=str, default = 'MIEC_generator', help="MIEC generator", required=False)
        parser.add_argument('--score_name', type=str, default = 'score_model', help="SF", required=False)
        parser.add_argument('--is_eval', type=bool, default =False, help='generate with same seed')
        parser.add_argument('--novel_check', type=bool, default =False, help='need the novel check path')
        parser.add_argument('--novel_check_path', type=str, default = 'traindata', help="the dataset to calculate novelty", required=False)
        parser.add_argument('--gensize', type=int, default = 5024, help="gensize,suppose to be the multiple of batchsize ", required=False)
        parser.add_argument('--batch_size', type=int, default = 32, help="batchsize", required=False)
        args = parser.parse_args()
        if args.is_eval:
            set_seed(42)
        else:
            set_seed((1,10000))
        savename=args.save_name
        with open('./dataset/require_seq_180.pkl','rb') as f:
            seqlist=pickle.load(f)
        AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
        stoi = { ch:i for i,ch in enumerate(AA_set) }
        itos = { i:ch for i,ch in enumerate(AA_set) }
        seqlistAA=[]
        for seq in seqlist:
            seqlistAA.append([stoi[i] for i in seq])
        mconf = RLpMIECConfig(36, 11,n_layer=8, num_miec=180,n_head=8, n_embd=256,is_pretrain=False,seqlist=seqlistAA)
        agent = RLpMIEC(mconf)
        agent.load_state_dict(torch.load(f'./model/{savename}.pt'))
        agent.to('cuda:0')
        agent.eval()
        prior = RLpMIEC(mconf)
        prior.load_state_dict(torch.load(f'./model/{args.prior_name}.pt'))
        prior.to('cuda:0')
        prior.eval()
        miec_conf = RLpMIECConfig(36, 11,num_miec=180,
                       n_layer=8, n_head=8, n_embd=256,seqlist=seqlistAA,is_mc=True)
        miec_model = RLpMIEC(miec_conf)
        miec_model.load_state_dict(torch.load('./model/'+args.miec_name+'.pt'))
        miec_model.to('cuda:0')
        myconf = myconfig()
        score_model = att_model(myconf)
        score_model.load_state_dict(torch.load('./model/'+args.score_name+'.pt'))
        score_model.eval()
        batch_size=args.batch_size
        block_size=12
        tmp=1
        count = 0
        sequence = []
        count += 1
        p=None
        miec_n=np.array([None])
        nllloss_n = np.array([None])
        pnllloss_n = np.array([None])
        cluster_n=np.array([None])
        score_n=np.array([None])
        gen_size=args.gensize
        gen_iter = math.ceil(gen_size / batch_size)
        set_seed(42)
        loss=[]
        t1= time.time()
        for i in tqdm(range(gen_iter)):
                p=None
                sca = None
                y,miec,cluster,nllloss =RL_sample(agent,miec_model,batch_size, block_size, begin=stoi['!'],temperature=tmp,sample=True) 
                t2= time.time()
                miec=miec.transpose(1,2).squeeze().cpu()
        
                nllloss=nllloss.squeeze().cpu()
                y=y.cpu()
                logits,_,_ = prior(y[:,:-1].to('cuda:0'),cluster = cluster.to('cuda:0'))
                score_device=next(score_model.parameters()).device
                enc = y[:,1:-1].detach().clone()
                for i in range(enc.shape[0]):
                    if enc[i,-1]==1:
                        enc[i,-1]=2
                clu = cluster-23
                out,_,_ = score_model(idx=enc.to(score_device),prop=miec[:,1:].to(score_device),cluster=clu.squeeze().to(score_device))
                score = out
                cluster=cluster.squeeze().cpu()
                targets=y[:,1:].detach().cpu()
                pnllloss=F.nll_loss(torch.log(F.softmax(logits[1].cpu(),-1).reshape(-1, logits[1].size(-1))), targets.reshape(-1),reduction='none').reshape(y.shape[0],-1).sum(-1)
                pnllloss=pnllloss.squeeze().cpu()
                for gen_mol in y:
                        completion = ''.join([itos[int(i)] for i in gen_mol])
                        completion=completion[1:] 
                        sequence.append((completion))
                if np.array(miec_n).all()==None:
                    miec_n=pd.DataFrame(np.array(miec.reshape(batch_size,-1).cpu()))
                    cluster_n=pd.DataFrame(np.array(cluster.cpu())-23)
                    nllloss_n=pd.DataFrame(np.array(nllloss.cpu()))
                    pnllloss_n=pd.DataFrame(np.array(pnllloss.detach().cpu()))
                    score_n=pd.DataFrame(score.detach().cpu().numpy())
                else:
                    miec_n=pd.concat([miec_n,pd.DataFrame(np.array(miec.reshape(batch_size,-1).cpu()))],axis=0).reset_index(drop=True)
                    cluster_n=pd.concat([cluster_n,pd.DataFrame(np.array(cluster.cpu())-23)],axis=0).reset_index(drop=True)
                    nllloss_n=pd.concat([nllloss_n,pd.DataFrame(np.array(nllloss.cpu()))],axis=0).reset_index(drop=True)
                    pnllloss_n=pd.concat([pnllloss_n,pd.DataFrame(np.array(pnllloss.detach().cpu()))],axis=0).reset_index(drop=True)
                    score_n=pd.concat([score_n,pd.DataFrame(score.cpu().detach().numpy())],axis=0).reset_index(drop=True)
                t3= time.time()
        seq_data=pd.concat([pd.DataFrame(sequence),cluster_n,miec_n,nllloss_n,pnllloss_n,score_n],axis=1)
        lenth=len(seq_data)
        illeg=[]
        for se in range(len(seq_data)):
            seq=seq_data.iloc[se,0]
        
            if '!' in seq[:]or '<' in seq[:-2]or '&' in seq[:-2]:
                illeg.append(se)
            seq_data.iloc[se,0]=seq[:-1].replace('<','').replace('&','')
        seq_data=seq_data.drop(index=illeg).reset_index(drop=True)

        eval_data = pd.DataFrame(np.array(seq_data))
        valid=len(eval_data)
        repetition=0
        re=len(eval_data)
        valid = len(eval_data)/args.gensize
        eval_data2=eval_data.drop_duplicates(([0,1])).reset_index(drop=True)
        repetition=re-len(eval_data2)
        uniq=len(eval_data2)/args.gensize
        dup_index=[]
        if args.novel_check:
            data=pd.read_csv(f'./dataset/{args.novel_check_path}.csv',header=None)
            for se in range(len(eval_data2)):
                seq=eval_data2.iloc[se,0]
                clu = eval_data2.iloc[se,1]
                if seq in np.array(data.iloc[:,1]):
                    dup_index.append((se,seq,clu))
            real_dup=[]
            recor = [] 
            for s in range(len(data)):
                seq=data.iloc[s,1]
                clu=data.iloc[s,0]
                for i,q,c in dup_index:
                    if seq==q and c==clu:
                        real_dup.append((i))
                        recor.append(s)
            eval_data3=eval_data2.drop(real_dup).reset_index(drop=True)
            novelty=len(eval_data3)/args.gensize
            eval_data3.to_csv(f'./output/{savename}_eval.csv',index=None)
            dup_data=eval_data2.iloc[real_dup,:].reset_index(drop=True)
            recor_data = data.iloc[recor,:].reset_index(drop=True)
        else:
            nocelty = 0
        c= 0
        for i in eval_data3.iloc[:,-1]:
            if i<=-9.562:
                c+=1
        with open(f'./output/{savename}_evaluation.log','w') as f:
            print(valid,uniq,novelty,repetition,len(real_dup),eval_data.iloc[:,-3].mean(),eval_data.iloc[:,-2].mean(),np.array(eval_data.iloc[:,-1]).mean(),c,sep='\n',file=f)
