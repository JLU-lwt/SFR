import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class IEEE754_Embedding(pl.LightningModule):
    def __init__(self, cfg):
        super(IEEE754_Embedding, self).__init__()
        self.bit32 = cfg.bit32
        self.norm = cfg.norm
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))

    def float2bit(self, f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
        ## SIGN BIT
        s = (torch.sign(f + 0.001) * -1 + 1) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[torch.isnan(e_scientific) == True] = -(2 ** (num_e_bits - 1) - 1)

        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1 / 2 ** (e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        if self.bit32:
            x = self.float2bit(x)
            x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
            x = x.view(x.shape[0], x.shape[1], -1)
            if self.norm:
                x = (x - 0.5) * 2
        
        return x



class NodeState_Embedding(pl.LightningModule):
    def __init__(self, cfg):
        super(NodeState_Embedding, self).__init__()
        self.ieee_embedding = IEEE754_Embedding(cfg)
        self.dim_hidden = cfg.dim_hidden
        self.ieee754 = cfg.ieee754


        self.embedding = nn.Sequential(
            nn.Linear(self.ieee754 * 3, self.dim_hidden),
            nn.LeakyReLU(inplace=True)
        )
        
        
    def forward(self, x):
        
        return self.embedding(self.ieee_embedding(x))

class NeighbourState_Embedding(pl.LightningModule):
    def __init__(self, cfg):
        super(NeighbourState_Embedding, self).__init__()
        self.cfg=cfg
        self.ieee_embedding = IEEE754_Embedding(cfg)
        self.dim_hidden = cfg.dim_hidden
        self.ieee754 = cfg.ieee754
        self.embedding_phi = nn.Sequential(
            nn.Linear(self.ieee754*3, cfg.dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            nn.LeakyReLU(inplace=True)
        )

        self.embedding_rho = nn.Sequential(
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cfg.dim_hidden, cfg.dim_hidden),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, node_state_emb,neighbour_state,info):

        neighbour_state=torch.scatter_add(torch.zeros_like(node_state_emb),
                          1,
                          info.unsqueeze(-1).repeat(1,1,self.cfg.dim_hidden).long(),
                          self.embedding_phi(self.ieee_embedding(neighbour_state))
                          )
        
        neighbour_state=self.embedding_rho(neighbour_state)
        
        return neighbour_state




class Y_Embedding(pl.LightningModule):
    def __init__(self, cfg):
        super(Y_Embedding, self).__init__()
        self.ieee_embedding = IEEE754_Embedding(cfg)
        self.ieee754 = cfg.ieee754
        self.dim_hidden = cfg.dim_hidden
        self.embedding = nn.Sequential(
            nn.Linear(self.ieee754, self.dim_hidden),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.embedding(self.ieee_embedding(x))

