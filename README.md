# LeaPformer
This repository contains the official implementation of "LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions," the preprint for which can be found [here](https://arxiv.org/abs/2405.13046). LeaPformers are, fundamentally, a novel modification of specific re-weighting functions for linear attention mechanisms that can enable them for a wider range of tasks. Due to improved flexibility, oftentimes LeaPformers are also more accurate than alternatives with only a small amount of added latency. 

Set-up for various parts of this repo are somewhat separated, as they were occasionally validated in different environments (i.e. the environment for LRA tests was not necessarily identical to the environment for LM or SimulST due to some compatibility issues). Instructions for set-up are provided in `pytorch-lra` and `fairseq-leapformer`.

## LeaPformers on the Long-Range Arena (LRA) Benchmark

Our slightly modified version of the [Skyformer](https://arxiv.org/abs/2111.00035) PyTorch LRA benchmark can be found in `pytorch-lra,` containing several additional linear attention mechanisms compared to the original implementation. Details for running the LRA benchmark are also provided there, including some example scripts.

As a note, this particular set-up focuses on extremely small models, allowing for tests with quadratic, softmax attention on long-sequence tasks for medium-to-low quality hardware. 

## LeaPformers on Autoregressive Language Modeling

We validated LeaPformers on small-scale autoregressive language modeling (i.e. around 140M parameters) via an older, private fork of Fairseq, provided in `fairseq-leapformer`. Scripts are available in `fairseq-leapformer/leapformer-scripts/lm` and, should one want to use a more updated version of Fairseq, it can be found [here](https://github.com/facebookresearch/fairseq).

## LeaPformers on S2T Simultaneous Translation (SimulST)

Similarly, we validated LeaPformers on SimulST on that same Fairseq fork. Unlike the autoregressive language modeling example, changes for SimulST are also placed in `fairseq-leapformer/examples/speech_to_text/simultaneous_translation/agents` and `fairseq-leapformer/examples/simultaneous_translation`, where some custom encoder-decoder masking occurs and the SimulEval agent is modified. Scripts are available in `fairseq-leapformer/leapformer-scripts/simulst`.

**Cleaning up. Will be finished soon.**

## What about more performant causal training/inference?

As mentioned in this work, our implementations (especially causal ones) are not optimized. A number of works have demonstrated the importance of constructing hardware-aware implementations to maximize performance. Obvious next steps here would be constructing a Triton-based LeaPformer implementation (à la [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) or FLA). In fact, integration with FLA is likely simple, especially for applications that are just decoder-based (e.g. autoregressive language modeling), requiring transforms being applied to the query and key before calling FLA specialized kernels.

## Other future steps for LeaPformers?

LeaPformers were originally conceived back in mid-2023, and a number of interesting works have been published since then containing elements which can be applied towards LeaPformers. For example: 

1. There are no RNN-like gating mechanisms in this work, despite concurrent work like [Gated Linear Attention](https://github.com/berlino/gated_linear_attention) (GLA) using it to great effect. 
2. Moreover, several works have skipped the time-dependent normalization term in linear attention, favoring normalization blocks (e.g. LayerNorm or GroupNorm, seen in papers [here](https://aclanthology.org/2022.emnlp-main.473/) and [here](https://arxiv.org/abs/2307.08621)), similarly seen in GLA. In our experiments, this made no real difference but might at scale.
3. Finally, the scale of the experiments in this work are ultimately small for modern applications, where it's very attractive to attempt to experiment at scale (i.e. around 300M+ minimum to several billion parameters).

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
