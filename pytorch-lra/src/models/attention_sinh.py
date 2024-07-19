import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinhLinearAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.device = config['device'] if 'device' in config else 'cuda'

        self.drop_attn = nn.Dropout(p = config["attention_dropout"])

    def unit_test(self, Q, K, V, mask):
        Q = F.relu(torch.sinh(Q))
        K = F.relu(torch.sinh(K))
      
        # checking equivalency, first linear implementation
        K_masked = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)
        
        # build out d x d intermediate matrices, then attn weights
        attn_inter = torch.matmul(K_masked.transpose(-2, -1), V)
        attn_weights = torch.matmul(Q, attn_inter)
        
        # build out normalization
        norm = torch.clamp_min(torch.matmul(Q, K_masked.transpose(-2, -1).sum(-1).unsqueeze(-1)), 1e-6)
        
        # final product for attn scores
        linear_attn = attn_weights / norm


        # then quadratic implementation
        dot = torch.matmul(Q, K.transpose(-2, -1))

        #print(dot)
        dot = dot.masked_fill(~(mask.to(bool)[:, None, None, :]), 0)
       
        denom = torch.clamp_min(torch.sum(dot, dim=-1).unsqueeze(-1), 1e-6)
        dot = dot / denom
        quad_attn = torch.matmul(dot, V)

        #linear_attn = torch.round(10000 * linear_attn) / 10000
        #quad_attn = torch.round(10000 * quad_attn) / 10000

        print(torch.eq(linear_attn[0, 0, :, :].sum(-1), quad_attn[0, 0, :, :].sum(-1)))
        print(linear_attn[0, 0, :, :].sum(-1))
        print(quad_attn[0, 0, :, :].sum(-1))
        print(torch.max(torch.abs(linear_attn - quad_attn)))
        assert 1 == 0

    def forward(self, Q, K, V, mask):
        
        # quick equivalency check
        #self.unit_test(Q, K, V, mask)

        Q = F.relu(torch.sinh(Q))
        K = F.relu(torch.sinh(K))

        # flip mask, then apply mask
        K = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)

        # build out d x d intermediate matrices, then attn weights
        attn_inter = torch.matmul(K.transpose(-2, -1), V)
        attn_weights = torch.matmul(Q, attn_inter)
        
        # build out normalization
        norm = torch.clamp_min(torch.matmul(Q, torch.abs(K).transpose(-2, -1).sum(-1).unsqueeze(-1)), 1e-6)
        
        # final product for attn scores
        attn = attn_weights / norm

        return attn
