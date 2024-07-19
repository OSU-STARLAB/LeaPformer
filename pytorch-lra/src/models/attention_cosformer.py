import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#import matplotlib.pyplot as plt
import torchplot as plt

class cosFormerAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.device = config['device'] if 'device' in config else 'cuda'

        self.drop_attn = nn.Dropout(p = config["attention_dropout"])

    def plot_props(self, prop1, prop2, bprop):
        prop1_temp = prop1[4, 0, :1024, :].repeat(1, 1024) 
        prop2_temp = prop2[4, 0, :1024, :].repeat(1, 1024)
        bprop_temp = bprop[:1024].unsqueeze(1).repeat(1, 1024) / 1024

        #cosformer_init_map = torch.cos((math.pi / 2) * (bprop_temp - bprop_temp.transpose(0, 1)))
        #img = plt.imshow(cosformer_init_map, cmap='hot', interpolation='nearest')
        #plt.colorbar(img)
        #plt.show()

        cosformer_learned_map = torch.cos((math.pi / 2) * (prop1_temp - prop2_temp.transpose(0, 1)))
        img = plt.imshow(cosformer_learned_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(img)
        plt.show()


    def forward(self, Q, K, V, mask):
        Q = F.relu(Q)
        K = F.relu(K)

        # apply mask
        #Q = Q.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)
        K = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)

        seq_len = Q.size(2)
        idx = torch.arange(1, seq_len + 1, device=self.device)

        # transform query and key into expanded form
        sin_tr = torch.sin((math.pi / 2) * (idx / seq_len))         
        cos_tr = torch.cos((math.pi / 2) * (idx / seq_len))
        q_sin = torch.mul(sin_tr.unsqueeze(-1), Q)
        q_cos = torch.mul(cos_tr.unsqueeze(-1), Q)

        k_sin = torch.mul(sin_tr, K.transpose(-2, -1))
        k_cos = torch.mul(cos_tr, K.transpose(-2, -1))


        # build out d x d intermediate matrices
        attn_inter_sin = torch.matmul(k_sin, V)
        attn_inter_cos = torch.matmul(k_cos, V)

        attn_weights_sin = torch.matmul(q_sin, attn_inter_sin)
        attn_weights_cos = torch.matmul(q_cos, attn_inter_cos)

        
        # build out normalization
        norm_sin = k_sin.sum(-1).unsqueeze(-1)
        norm_cos = k_cos.sum(-1).unsqueeze(-1)

        norm = torch.matmul(q_sin, norm_sin) + torch.matmul(q_cos, norm_cos)

        # final product for attn scores
        attn = (attn_weights_sin + attn_weights_cos) / norm

        return attn
