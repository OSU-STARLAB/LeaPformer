# SimulST Script Overview

The scripts provided in this directory are really just examples, you should tweak them as necessary according to your needs/hardware constraints. This small doc is here to explain a few quirks here, as opposed to language modeling where the flow is short and extremely obvious.

The general flow of usage is the following:

```
Preprocessing => ASR Pretraining => SimulST Training => Eval Data Split => Eval
```

### LeaPformer Usage and Customizability

By default, all training scripts are configured to replace all attention blocks with equivalent LeaPformer blocks. Disabling this replacement for a given block is as simple as removing its flag (marked pretty literally, e.g. `--enc-leapformer-enable` of course corresponds to encoder self-attention). Changing the step-down factor requires changing the value passed to `--leap-factor`, which is set to 4 by default. 

Linearized training is disabled by default and isn't recommended without custom kernels due to constraining memory requirements for mostly parallelizable implementations.

> [!WARNING]  
> You should not change encoder settings when loading a checkpoint after ASR pretraining. This can result in sub-par model performance, poorer convergence, and generally unintended results.

> [!WARNING]  
> Currently running into some problems in reproducing models of the the same level of quality as in the paper. Seems to be an environmental issue as baseline models are affected as well. 
