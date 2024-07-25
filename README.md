# LeaPformer
This repository contains the official implementation of "LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions," the preprint for which can be found [here](https://arxiv.org/abs/2405.13046). LeaPformers are, fundamentally, a novel modification of specific re-weighting functions for linear attention mechanisms that can enable them for a wider range of tasks. Due to improved flexibility, oftentimes LeaPformers are also more accurate than alternatives with only a small amount of added latency. More details will be provided here, soon.

## LeaPformers on the Long-Range Arena (LRA) Benchmark

Our slightly modified version of the [Skyformer](https://arxiv.org/abs/2111.00035) PyTorch LRA benchmark can be found in `pytorch-lra,` containing several additional linear attention mechanisms compared to the original implementation. Details for running the LRA benchmark are also provided there, including some example scripts.

As a note, this particular set-up focuses on extremely small models, allowing for tests with quadratic, softmax attention on long-sequence tasks for medium-to-low quality hardware. 

## LeaPformers on Autoregressive Language Modeling

We validated LeaPformers on small-scale autoregressive language modeling (i.e. around 140M parameters) via an older fork of Fairseq, to be provided in `fairseq-leapformer` (still being cleaned up, initial implementation was ad-hoc). 

**Cleaning up. Will be finished soon.**

## LeaPformers on S2T Simultaneous Translation (SimulST)

We validated LeaPformers on SimulST via an older fork of Fairseq, to be provided in `fairseq-leapformer` (still being cleaned up, initial implementation was ad-hoc). Unlike the autoregressive language modeling example, changes for SimulST are also placed in `fairseq-leapformer/examples/speech_to_text/simultaneous_translation/agents` and `fairseq-leapformer/examples/simultaneous_translation`, where some custom encoder-decoder masking occurs and the SimulEval agent is modified.

**Cleaning up. Will be finished soon.**

## What about more performant causal training/inference?

As mentioned in this work, our implementations (especially causal ones) are not optimized. A number of works have demonstrated the importance of constructing hardware-aware implementations to maximize performance. Obvious next steps here would be constructing a Triton-based LeaPformer implementation (à la [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention)). In fact, integration with Flash Linear Attention (FLA) is likely simple, especially for applications that are just decoder-based (e.g. autoregressive language modeling), requiring transforms being applied to the query and key before calling FLA specialized kernels.

## Reference

If you found our work insightful or useful, please consider citing us as:

```
@inproceedings{
      agostinelli2024leapformer,
      title={LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions},
      author={Victor Agostinelli and Sanghyun Hong and Lizhong Chen},
      booktitle={Forty-first International Conference on Machine Learning},
      year={2024},
      url={https://openreview.net/forum?id=XhH1OKLANY}
}
```
