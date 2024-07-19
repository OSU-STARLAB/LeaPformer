# LeaPformer
This repository contains the official implementation of "LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions," the preprint for which can be found [here](https://arxiv.org/abs/2405.13046). LeaPformers are, fundamentally, a novel modification of specific re-weighting functions for linear attention mechanisms that can enable them for a wider range of tasks. Due to improved flexibility, oftentimes LeaPformers are also more accurate than alternatives with only a small amount of added latency. More details will be provided here, soon.

## LeaPformers on the Long-Range Arena (LRA) Benchmark

Our slightly modified version of the [Skyformer](https://arxiv.org/abs/2111.00035) PyTorch LRA benchmark can be found in `pytorch-lra,` containing several additional linear attention mechanisms compared to the original implementation. Details for running the LRA benchmark are also provided there, including some example scripts.

## LeaPformers on Autoregressive Language Modeling

TODO

## LeaPformers on S2T Simultaneous Translation (SimulST)

TODO

## Reference

If you found our work insightful or useful, please consider citing us as:

```
@misc{agostinelli2024leapformerenablinglineartransformers,
      title={LeaPformer: Enabling Linear Transformers for Autoregressive and Simultaneous Tasks via Learned Proportions}, 
      author={Victor Agostinelli and Sanghyun Hong and Lizhong Chen},
      year={2024},
      eprint={2405.13046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.13046}, 
}
```
