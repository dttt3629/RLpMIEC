
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import pdb
time =0
logger = logging.getLogger(__name__)
class RLpMIECConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    def __init__(self, vocab_size, block_size, is_pretrain=False,seqlist=None,is_mc=False,**kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.is_pretrain = is_pretrain
        self.seqlist = seqlist
        self.is_mc = is_mc
        for k,v in kwargs.items():
            setattr(self, k, v)
class BidSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        num = 1
        martrix=torch.tril(torch.ones(config.block_size+config.num_miec+num,config.block_size+config.num_miec+num)).view(1, 1, config.block_size+config.num_miec+num, config.block_size+config.num_miec+num)
        martrix[:,:,:,config.block_size+num+1:]=1
        martrix[:,:,0,1:]=0
        self.register_buffer("mask", martrix)
        self.n_head = config.n_head

    def forward(self, x, layer_past=None,valid_len=None):
        B, T, C = x.size()
        global time
        k= self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y, attn_save
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
        num = 1
        martrix=torch.tril(torch.ones(config.block_size+config.num_miec+num,config.block_size+config.num_miec+num)).view(1, 1, config.block_size+config.num_miec+num, config.block_size+config.num_miec+num)
        martrix[:,:,config.block_size+num+1:,config.block_size+num+1:]=1
        self.register_buffer("mask", martrix)
        self.n_head = config.n_head

    def forward(self, x, layer_past=None,valid_len=None):
        B, T, C = x.size()
        global time
        k= self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y, attn_save
class BidBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = BidSelfAttention(config)
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
        # pdb.set_trace()
        return x, attn

class RLpMIEC(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(3, config.n_embd)
        self.miec_nn = nn.Linear(config.num_miec, config.n_embd) #no use but included in the current model
        self.pos_emb = nn.Parameter(torch.randn(1, 12+180,config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.bidblock = BidBlock(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer-1)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size , bias=False)
        self.linein2 = nn.Linear(config.vocab_size ,4,bias=True)
        self.block_size = config.block_size
        self.is_pretrain=config.is_pretrain
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 

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
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    def enc(self,idx):
        b, t = idx.size()
        x = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, 1:t+1, :] #use for T-SNE
        return x+position_embeddings
    def forward(self, idx, targets=None, prop = None,valid_len=None,cluster=None):
        b, t = idx.size()
        rec_len=len(self.config.seqlist[0])
        if t<11:
            idx=torch.cat([idx,torch.zeros(b,(11-t),dtype=idx.dtype,device=idx.device)],1)
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector (b,20,256)
        position_embeddings = self.pos_emb[:, :12+rec_len, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,11), dtype = torch.long, device = idx.device))
        type_embeddings2 = self.type_emb(torch.ones((b,rec_len), dtype = torch.long, device = idx.device)*2)
        x = token_embeddings  + type_embeddings
        con_typ=self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
        if cluster == None:
            cluster=torch.zeros((b,1),dtype=idx.dtype,device=idx.device)+23
        con_tok=self.tok_emb(cluster)
        con=con_typ+con_tok
        recpseq=torch.tensor([self.config.seqlist[ind-23] for ind in cluster],device=idx.device)
        x = torch.cat([x[:,:1,:],con,x[:,1:,:]],1)
        recpseq = self.tok_emb(recpseq)
        recpseq=self.drop(recpseq  + type_embeddings2)
        x=self.drop(torch.cat([x,recpseq],1)+position_embeddings)
        attn_maps = []
        # print(x,x.shape)
        x,attn=self.bidblock(x)
        attn_maps.append(attn)
        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)
        x = self.ln_f(x)
        logits=self.head(x)
        seq=logits[:,1:12,:23]
        clu=logits[:,:1,23:]
        out=logits[:,:12,:]
        logits = self.linein2(logits)
        miec=logits[:,12:,:]
        loss = None
        alpha=1
        bite=1
        loss_a=None
        loss_b=None
        loss_c = None
        if targets is not None:
            loss_b= bite*F.mse_loss(miec.squeeze().transpose(1,2),prop.squeeze(),reduction='none').mean(dim=-1).mean(-1)
            loss_c = F.mse_loss(miec.transpose(1,2).sum(1),prop.sum(1),reduction='none').mean(-1).reshape(b,-1)
            loss_a =alpha* F.cross_entropy(out.reshape(-1, out.size(-1)), targets.view(-1),reduction='none').reshape(b,-1).mean(-1)
            if self.config.is_pretrain:
                loss = loss_a
            else:
                if self.config.is_mc:
                    loss = loss_b
                else:
                    loss=loss_a
        return (miec,seq,clu), (loss,loss_a,loss_b,loss_c), attn_maps