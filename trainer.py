
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
logger = logging.getLogger(__name__)
def try_gpu(i=0): 
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus(): 
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
class TrainerConfig:
    batch_size = 64
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 
    lr_decay = False
    warmup_tokens = 375e6 
    final_tokens = 260e9 
    ckpt_path = None
    num_workers = 4
    epoch_rcd =[]
    train_l_rcd=[]
    train_la_rcd=[]
    train_lb_rcd=[]
    train_lc_rcd=[]
    val_l_rcd=[]
    val_la_rcd=[]
    val_lb_rcd=[]
    val_lc_rcd=[]
    avs_l_rcd=[]
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset,val_dataset, config, stoi, itos,is_pretrain=False):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.val_dataset = val_dataset
        self.stoi = stoi
        self.itos = itos
        self.is_pretrain=is_pretrain
        num_gpus=4
        if torch.cuda.is_available():
            ##for all gpu
            #self.devices = [try_gpu(i) for i in range(num_gpus)]

            #self.model = torch.nn.DataParallel(self.model,device_ids=self.devices).to(self.device)
            # For single gpu
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        def run_epoch(split):
            is_train = split == 'train'
            is_val = split =='val'
            model.train(is_train)
            if is_train:
                data=self.train_dataset
            elif is_val:
                data=self.val_dataset
            else:
                data=self.val_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            average_smi=[]
            same=[]
            losses_a=[]
            losses_b=[]
            losses_c=[]
            for it, (x, y, p,valen,clu) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                p = p.to(self.device)
                valen = valen.to(self.device)
                clu = clu.to(self.device)
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(x, y, p,valid_len=valen,cluster=clu)
                        if not self.is_pretrain:
                            loss_a=loss[1].mean()
                            loss_b=loss[2].mean()
                            losses_a.append(loss_a.item())
                            losses_b.append(loss_b.item())
                            loss_c = loss[3].mean()
                            losses_c.append(loss_c.item())
                        loss= loss[0].mean() # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                # forward the model
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)-progress/4)) #-progress to aceelerate decay
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    # wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                if is_val:
                    out=logits[1]
                    out=F.softmax(out,dim=-1)
                    value,indice=torch.topk(out,1)
                    indice=indice.squeeze()
                    smil=[]
                    for i in range(indice.shape[0]):
                        smi=[]
                        for s in range(indice.shape[-1]):
                            if s+1<y.shape[-1]:
                                if indice[i,s].item()==y[i,s+1].item():
                                    smi.append(1)
                                else:
                                    smi.append(0)
                            else:
                                smi.append(0)
                        smil.append(sum(smi)/len(smi))
                        if sum(smi)==len(smi):
                            same.append(1)
                        else:
                            same.append(0)
                    average_smi.append(sum(smil)/len(smil))

  
            if is_train:
                if not self.is_pretrain:
                    return float(np.mean(losses)),float(np.mean(losses_a)),float(np.mean(losses_b)),float(np.mean(losses_c))
                return float(np.mean(losses))

            if not is_train:
                if not is_val:
                    test_loss = float(np.mean(losses))
                    logger.info("test loss: %f", test_loss)
                    if not self.is_pretrain:
                        return test_loss,float(np.mean(losses_a)),float(np.mean(losses_b)),float(np.mean(losses_c))
                    return test_loss
                else:
                    epoch_smi=sum(average_smi)/len(average_smi)
                    epoch_same=sum(same)
                    return epoch_smi
                    
                    

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []

        for epoch in range(config.max_epochs):
            if not self.is_pretrain:
                train_loss,train_loss_a,train_loss_b ,train_loss_c= run_epoch('train')
            else:
                train_loss = run_epoch('train')
            if not self.is_pretrain: #加速
                if self.val_dataset is not None:
                    smi=run_epoch('val')
            if self.test_dataset is not None:
                if not self.is_pretrain:
                    test_loss,test_loss_a,test_loss_b,test_loss_c = run_epoch('test')
                else:
                    test_loss = run_epoch('test')
            if self.is_pretrain:

                config.epoch_rcd.append(epoch+1)
                config.train_l_rcd.append(train_loss)
                config.val_l_rcd.append(test_loss)
                pd.DataFrame([config.epoch_rcd,config.train_l_rcd,config.val_l_rcd]).T.to_csv(f'./output/{self.config.ckpt_path}_pre_result.csv')

            else:
                config.epoch_rcd.append(epoch+1)
                config.train_l_rcd.append(train_loss)
                config.train_la_rcd.append(train_loss_a)
                config.train_lb_rcd.append(train_loss_b)
                config.train_lc_rcd.append(train_loss_c)
                config.val_l_rcd.append(test_loss)
                config.val_la_rcd.append(test_loss_a)
                config.val_lb_rcd.append(test_loss_b)
                config.val_lc_rcd.append(test_loss_c)
                config.avs_l_rcd.append(smi)
                pd.DataFrame([config.epoch_rcd,config.train_l_rcd,config.train_la_rcd,config.train_lb_rcd,config.train_lc_rcd, config.val_l_rcd,config.val_la_rcd,config.val_lb_rcd,config.val_lc_rcd, config.avs_l_rcd]).T.to_csv(f'./result/{self.config.ckpt_path}_result.csv')



 
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.val_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()
        return None
