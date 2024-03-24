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

        self.enable_learned_numer = config["enable_learned_numer"]
        self.learned_numer_inter_size = config["learned_numer_inter_size"]
        print(f"Learned proportions are set to {self.enable_learned_numer}")
        if self.enable_learned_numer:
            #self.q_numer =  nn.Linear(self.head_dim, 1, bias=False) 
            #self.k_numer =  nn.Linear(self.head_dim, 1, bias=False) 

            if self.learned_numer_inter_size == 1:
                self.q_numer = nn.Sequential(
                        nn.Linear(self.head_dim, 1),
                        nn.Sigmoid()
                )
                self.k_numer = nn.Sequential(
                        nn.Linear(self.head_dim, 1),
                        nn.Sigmoid()
                )
            
            else:
                self.q_numer = nn.Sequential(
                        nn.Linear(self.head_dim, self.learned_numer_inter_size),
                        nn.ReLU(),
                        nn.Linear(self.learned_numer_inter_size, 1),
                        nn.Sigmoid()
                )
                self.k_numer = nn.Sequential(
                        nn.Linear(self.head_dim, self.learned_numer_inter_size),
                        nn.ReLU(),
                        nn.Linear(self.learned_numer_inter_size, 1),
                        nn.Sigmoid()
                )

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
        if self.enable_learned_numer:
            q_num = self.q_numer(Q)
            k_num = self.k_numer(K)
            
            q_sin_tr = torch.sin((math.pi / 2) * q_num)         
            q_cos_tr = torch.cos((math.pi / 2) * q_num)
            q_sin = torch.mul(q_sin_tr, Q)
            q_cos = torch.mul(q_cos_tr, Q)
            
            #k_sin_tr = torch.sin((math.pi / 2) * (idx / seq_len))         
            #k_cos_tr = torch.cos((math.pi / 2) * (idx / seq_len))
            #k_sin = torch.mul(k_sin_tr, K.transpose(-2, -1))
            #k_cos = torch.mul(k_cos_tr, K.transpose(-2, -1))
            
            k_sin_tr = torch.sin((math.pi / 2) * k_num)         
            k_cos_tr = torch.cos((math.pi / 2) * k_num)
            k_sin = torch.mul(k_sin_tr.transpose(-2, -1), K.transpose(-2, -1))
            k_cos = torch.mul(k_cos_tr.transpose(-2, -1), K.transpose(-2, -1))

        else:
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
