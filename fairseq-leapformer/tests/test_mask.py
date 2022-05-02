
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--encoder-mask-future-delay",
    type=int,
    metavar="N",
    help="Specify degree of attention into future when using encoder attention masking. Default=infinite (full attention)",
)
parser.add_argument(
    "--encoder-mask-block-size",
    type=int,
    metavar="N",
    help="Specify size of input blocks for which block attention is allowed. Useful when inputs will be grouped following encoder (e.g., simultaneous translation using wait-k). Default=1 (no additional attention)",
)
args = parser.parse_args()

delay = getattr(args, "encoder_mask_future_delay")
if delay is None:
    delay = float('inf')
block_size = getattr(args, "encoder_mask_block_size", 1)
if block_size is None:
    block_size = 1

#print(delay)
#print(block_size)
dim = 6

if (delay >= dim-1): # Full attention allowed, no need to check other conditions
    future_mask = torch.zeros([dim, dim])
else:
    tri_mask = torch.triu(   # Start with mask that disallows looking into future
        torch.full((dim,dim), -999), 1
    )
    # Create additional masks that consider self.encoder_mask_future_delay and self.encoder_block_size
    block_count = dim // block_size
    block_pad = dim % block_size
    blocks = torch.full((block_count, block_size, block_size), 1, dtype=torch.bool)
    block_mask = torch.nn.functional.pad(input=torch.block_diag(*blocks), pad=(0, block_pad, 0, block_pad))

    delay_mask = torch.cat(
        (
            torch.full((dim,delay+1), 1, dtype=torch.bool),
            torch.zeros((dim,dim-(delay+1)), dtype=torch.bool)
        ), 1
    )
    
    # VA, covers edge case where dim is less than block_size and the block_mask logic is a dimension off
    if dim < block_size:
        block_mask = block_mask[:-1]

    corr_mask = torch.logical_or(block_mask, delay_mask)

    print(f"Block mask:\n{block_mask}")
    print(f"Delay mask:\n{delay_mask}")
    print(f"Corr mask:\n{corr_mask}")

    future_mask = tri_mask.masked_fill_(corr_mask, 0) # Apply correction

print(f"Future mask:\n{future_mask}")
