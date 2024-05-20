
import numpy as np 
import pandas as pd 
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import math
from tqdm import tqdm
def set_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %%

class mydataset(Dataset):
    def __init__(self,data,stoi,aug_prob = 0.5):
        super().__init__()
        self.seq=np.array(data.iloc[:,1])
        self.label = torch.tensor(np.array(data.iloc[:,2]),dtype=torch.float32)
        self.stoi = stoi
        self.cluster = torch.tensor(np.array(data.iloc[:,0]))
        self.miec1 = torch.tensor(np.array(data.iloc[:,3::4]),dtype=torch.float32)
        self.miec2 = torch.tensor(np.array(data.iloc[:,4::4]),dtype=torch.float32)
        self.miec3 = torch.tensor(np.array(data.iloc[:,5::4]),dtype=torch.float32)
        self.miec4 = torch.tensor(np.array(data.iloc[:,6::4]),dtype=torch.float32)
        self.aug_prob = aug_prob 
        #self.block_size = block_size
    def __len__(self):
        return len(self.seq)
    def randomize_aaseq(self,aaseq):
        AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
        aa_len=len(aaseq)
        random_seq=''
        for i in range(aa_len):
            random_seq+=np.random.choice(AA_set)
        return random_seq
    def __getitem__(self, idx):
        # p = np.random.uniform()
        seq = self.seq[idx]
        # if p < self.aug_prob:
        #     seq=self.randomize_aaseq(seq)
        seq = [self.stoi[i] for i in seq]
        # miec1 = self.miec1[idx][require]
        # miec2 = self.miec2[idx][require]
        # miec3 = self.miec3[idx][require]
        # miec4 = self.miec4[idx][require]
        miec1 = self.miec1[idx]
        miec2 = self.miec2[idx]
        miec3 = self.miec3[idx]
        miec4 = self.miec4[idx]
        miec = torch.concat([miec1,miec2,miec3,miec4]).reshape(4,-1)
        cluster = self.cluster[idx]
        label = self.label[idx]
        if len(seq)<10:
            seq .append(stoi['<'])
        return torch.tensor(seq),label,cluster,miec


class myconfig:
    def __init__(self, n_head=8, n_embd=256, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop = 0.1):
            self.n_embd = n_embd
            self.attn_pdrop = attn_pdrop
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.n_head = n_head
            self.vocab_size= 23
            self.num_miec = 180
            self.n_layer = 4
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        global time
        k= self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y, attn_save
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn
class att_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.clu_emb = nn.Embedding(13, config.n_embd)
        self.miec_nn = nn.Linear(config.num_miec, config.n_embd)
        self.miec_nn2 = nn.Linear(config.num_miec, config.n_embd)
        self.miec_nn3 = nn.Linear(config.num_miec, config.n_embd)
        self.miec_nn4 = nn.Linear(config.num_miec, config.n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, 15,config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1 , bias=False)
        self.linein = nn.Linear(config.vocab_size ,1,bias=True)
        # self.block_size = config.block_size
        self.line2 = nn.Linear(14*16,1)
        self.flat = nn.Flatten()
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.apply(self._init_weights)
       

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        decay.add('pos_emb')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))]},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate)
        return optimizer
    def enc(self,idx):
        b, t = idx.size()
        x = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, 1:t+1, :]
        return x+position_embeddings
    def forward(self, idx, targets=None, prop = None,cluster=None):
            b, t = idx.size()
            token_embeddings = self.tok_emb(idx) 
            position_embeddings = self.pos_emb[:, :, :] # each position maps to a (learnable) vector
            type_embeddings = self.type_emb(torch.ones((b,10), dtype = torch.long, device = idx.device))
            # position_embeddings2 = self.pos_emb2[:, :, :] # each position maps to a (learnable) vector
            #pdb.set_trace()
            x = token_embeddings  + type_embeddings
        
            con_typ=self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            con_tok=self.clu_emb(cluster.reshape(b,1))
            
            con=con_typ+con_tok
            # pdb.set_trace()
            #miec = self.miec_nn(prop[:,0]).reshape(b,1,-1)
            miec2 = self.miec_nn2(prop[:,0]).reshape(b,1,-1)
            miec3 = self.miec_nn3(prop[:,1]).reshape(b,1,-1)
            miec4 = self.miec_nn4(prop[:,2]).reshape(b,1,-1)

            x = torch.cat([con,x,miec2,miec3,miec4],1)
            
            x=self.drop(x+position_embeddings[:,:-1,:])
            attn_maps = []
            # print(x,x.shape)
            for layer in self.blocks:
                x, attn = layer(x)
                attn_maps.append(attn)
            x = self.ln_f(x)
            logits=self.head(x)
            out = self.line2(self.flat(logits))
            loss = None
            if targets is not None:
                loss = F.mse_loss(out.squeeze(),targets.squeeze())
            return out,loss,attn_maps
    

if __name__=='__main__':
    set_seed(42)
    data=pd.read_csv('./dataset/mhc_data_full_cleanover2.csv')
    data= data.sample(frac = 1).reset_index(drop=True)  #randomlised
    num_data=data.shape[0]
    train_data=data.iloc[:int(num_data*0.8),:].copy().reset_index(drop=True)
    test_data=data.iloc[int(num_data*0.8):int(num_data*0.9),:].copy().reset_index(drop=True)
    val_data=data.iloc[int(num_data*0.9):,:].copy().reset_index(drop=True)
    AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&'])
    stoi = { ch:i for i,ch in enumerate(AA_set) }
    itos = { i:ch for i,ch in enumerate(AA_set) }
    batch_size=64
    lr = 4e-4
    max_epoch=200
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    config = myconfig()
    config.learning_rate = lr
    set_seed(42)
    model=att_model(config).to(device)
    save_name = './model/score_model.pt'
    train_dataset = mydataset(train_data,stoi,aug_prob=0)
    val_dataset = mydataset(val_data,stoi,aug_prob=0)
    test_dataset = mydataset(test_data,stoi,aug_prob=0)
    best_loss=None
    train_losses=[]
    test_losses = []
    optimizer = model.configure_optimizers(config)
    config.progress = 0
    for epoch in range(max_epoch):
        def train_epoch(model,dataset,batch_size,is_train):
                loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                                batch_size=batch_size)
                pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
                losses=[]
                model.train(is_train)
                com =[]
                re=[]

                for it,(x,y,clu,miec) in pbar:
                    x=x.to(device)
                    y=y.to(device)
                    miec=miec.to(device)
                    clu = clu.to(device)
                    logits,loss,_=model(x,y,prop=miec[:,1:],cluster=clu)
                    loss=loss.mean()
                    losses.append(loss.item())
                    re.extend(y.cpu().tolist())
                    com.extend(logits[:,-1,0].cpu().tolist())

                    if is_train:
                        config.progress+=float(1/(len(loader)*max_epoch))
                        lr_mult = max(0.05, 0.5 * (1.0 + math.cos(math.pi * config.progress)-config.progress/4)) 
                        lr = config.learning_rate * lr_mult
                        optimizer = model.configure_optimizers(config)
                        for param_group in optimizer.param_groups:
                                    param_group['lr'] = lr
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {optimizer.param_groups[0]['lr']}")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                return np.mean(losses)
        train_loss= train_epoch(model,train_dataset,batch_size,is_train=True)
        test_loss = train_epoch(model,val_dataset,batch_size,is_train=False)    
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if best_loss is not None:
            if test_loss >best_loss :
                best_loss = test_loss
                print(f'save at{epoch}:{best_loss}')
                torch.save(model.state_dict(), save_name)
        else:
            best_loss=test_loss