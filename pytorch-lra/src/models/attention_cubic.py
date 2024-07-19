import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CubicLinearAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.device = config['device'] if 'device' in config else 'cuda'

        self.drop_attn = nn.Dropout(p = config["attention_dropout"])


    def forward(self, Q, K, V, mask):
        # experimental, removing ReLU to start
        #Q = F.relu(Q)
        #K = F.relu(K)

        # apply mask
        K = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)
        
        Q = torch.pow(Q, 3)
        K = torch.pow(K, 3)

        # build out d x d intermediate matrices, then attn weights
        attn_inter = torch.matmul(K.transpose(-2, -1), V)
        attn_weights = torch.matmul(Q, attn_inter)
        
        # build out normalization
        norm = torch.matmul(Q, K.transpose(-2, -1).sum(-1).unsqueeze(-1))
        
        # final product for attn scores
        attn = attn_weights / norm

        return attn
