"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes.
   Taken directly from the S4 repository at https://github.com/state-spaces/s4/tree/main
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.model_transformer import Embeddings

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=0.00001):
        super().__init__()
        # Generate dt
        H = d_model
        self.log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        #self.register("log_dt", log_dt, lr)

        self.log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        self.A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        #self.register("log_A_real", log_A_real, lr)
        #self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # weird device sync
        dt = dt.to(device="cuda")
        C = C.to(device="cuda")
        A = A.to(device="cuda")

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        #self.transposed = transposed
        self.transposed = False

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        #dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, self.h, kernel_size=1),
            nn.ReLU(),
            #nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            #nn.GLU(dim=-2),
        )

    def forward(self, u, mask): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        u = u.to(dtype=torch.float32)
        u = u * mask

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L).to(device=u.device) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y

# added from Skyformer LRA implementation
class S4Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_layers = config["num_layers"]
        
        self.embeddings = Embeddings(config)

        for idx in range(self.num_layers):
            setattr(self, f"s4_{idx}", S4D(d_model=config["transformer_dim"], dropout=config["dropout_prob"]))


    def forward(self, input_ids, mask = None):

        X = self.embeddings(input_ids)

        if mask is None:
            mask = torch.ones_like(input_ids)
        
        # covers edge case for S4
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        for idx in range(self.num_layers):
            X = getattr(self, f"s4_{idx}")(X, mask)

        return X
