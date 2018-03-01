# Introduction
FAIR Sequence-to-Sequence Toolkit (PyTorch)

This is a PyTorch version of [fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence learning toolkit from Facebook AI Research. The original authors of this reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam Gross. The toolkit implements the fully convolutional model described in [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122) and features multi-GPU training on a single machine as well as fast beam search generation on both CPU and GPU. We provide pre-trained models for English to French and English to German translation.

![Model](fairseq.gif)

# Citation

If you use the code in your paper, then please cite it as:

```
@inproceedings{gehring2017convs2s,
  author    = {Gehring, Jonas, and Auli, Michael and Grangier, David and Yarats, Denis and Dauphin, Yann N},
  title     = "{Convolutional Sequence to Sequence Learning}",
  booktitle = {Proc. of ICML},
  year      = 2017,
}
```

# Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

Currently fairseq-py requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size` as command line
options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq-py with:
```
pip install -r requirements.txt
python setup.py build
python setup.py develop
```

# Quick Start

The following command-line tools are available:
* `python preprocess.py`: Data pre-processing: build vocabularies and binarize training data
* `python train.py`: Train a new model on one or multiple GPUs
* `python generate.py`: Translate pre-processed data with a trained model
* `python interactive.py`: Translate raw text with a trained model
* `python score.py`: BLEU scoring of generated translations against reference translations

## Evaluating Pre-trained Models
First, download a pre-trained model along with its vocabularies:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf -
```

This model uses a [Byte Pair Encoding (BPE) vocabulary](https://arxiv.org/abs/1508.07909), so we'll have to apply the encoding to the source text before it can be translated.
This can be done with the [apply_bpe.py](https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py) script using the `wmt14.en-fr.fconv-cuda/bpecodes` file.
`@@` is used as a continuation marker and the original text can be easily recovered with e.g. `sed s/@@ //g` or by passing the `--remove-bpe` flag to `generate.py`.
Prior to BPE, input text needs to be tokenized using `tokenizer.perl` from [mosesdecoder](https://github.com/moses-smt/mosesdecoder).

Let's use `python interactive.py` to generate translations interactively.
Here, we use a beam size of 5:
```
$ MODEL_DIR=wmt14.en-fr.fconv-py
$ python interactive.py \
 --path $MODEL_DIR/model.pt $MODEL_DIR \
 --beam 5
| loading model(s) from wmt14.en-fr.fconv-py/model.pt
| [en] dictionary: 44206 types
| [fr] dictionary: 44463 types
| Type the input sentence and press return:
> Why is it rare to discover new marine mam@@ mal species ?
O       Why is it rare to discover new marine mam@@ mal species ?
H       -0.06429661810398102    Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins ?
A       0 1 3 3 5 6 6 8 8 8 7 11 12
```

This generation script produces four types of outputs: a line prefixed with *S* shows the supplied source sentence after applying the vocabulary; *O* is a copy of the original source sentence; *H* is the hypothesis along with an average log-likelihood; and *A* is the attention maxima for each word in the hypothesis, including the end-of-sentence marker which is omitted from the text.

Check [below](#pre-trained-models) for a full list of pre-trained models available.


## Training a New Model

### Data Pre-processing
The fairseq-py source distribution contains an example pre-processing script for
the IWSLT 2014 German-English corpus.
Pre-process and binarize the data as follows:
```
$ cd data/
$ bash prepare-iwslt14.sh
$ cd ..
$ TEXT=data/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en
```
This will write binarized data that can be used for model training to `data-bin/iwslt14.tokenized.de-en`.

### Training
Use `python train.py` to train a new model.
Here a few example settings that work well for the IWSLT 2014 dataset:
```
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 python train.py data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

By default, `python train.py` will use all available GPUs on your machine.
Use the [CUDA_VISIBLE_DEVICES](http://acceleware.com/blog/cudavisibledevices-masking-gpus) environment variable to select specific GPUs and/or to change the number of GPU devices that will be used.

Also note that the batch size is specified in terms of the maximum number of tokens per batch (`--max-tokens`).
You may need to use a smaller value depending on the available GPU memory on your system.

### Generation
Once your model is trained, you can generate translations using `python generate.py` **(for binarized data)** or `python interactive.py` **(for raw text)**:
```
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/fconv/checkpoint_best.pt \
  --batch-size 128 --beam 5
  | [de] dictionary: 35475 types
  | [en] dictionary: 24739 types
  | data-bin/iwslt14.tokenized.de-en test 6750 examples
  | model fconv
  | loaded checkpoint trainings/fconv/checkpoint_best.pt
  S-721   danke .
  T-721   thank you .
  ...
```

To generate translations with only a CPU, use the `--cpu` flag.
BPE continuation markers can be removed with the `--remove-bpe` flag.

# Pre-trained Models

We provide the following pre-trained fully convolutional sequence-to-sequence models:

* [wmt14.en-fr.fconv-py.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) including vocabularies
* [wmt14.en-de.fconv-py.tar.bz2](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-de.fconv-py.tar.bz2): Pre-trained model for [WMT14 English-German](https://nlp.stanford.edu/projects/nmt) including vocabularies

In addition, we provide pre-processed and binarized test sets for the models above:
* [wmt14.en-fr.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-French
* [wmt14.en-fr.ntst1213.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.ntst1213.tar.bz2): newstest2012 and newstest2013 test sets for WMT14 English-French
* [wmt14.en-de.newstest2014.tar.bz2](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-de.newstest2014.tar.bz2): newstest2014 test set for WMT14 English-German

Generation with the binarized test sets can be run in batch mode as follows, e.g. for English-French on a GTX-1080ti:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Scoring with score.py:
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

# Distributed version

Distributed training in fairseq-py is implemented on top of [torch.distributed](http://pytorch.org/docs/master/distributed.html).
In order to run it requires one process per GPU. In order for those processes to be able to discover each other
they need to know a unique host and port that can be used to establish initial connection and each process
needs to be assigned a rank, that is a unique number from 0 to n-1 where n is the total number of GPUs.

Below is the example of training of a big En2Fr model on 16 nodes with 8 GPUs each (in total 128 GPUs):

If you run on a cluster managed by [SLURM](https://slurm.schedmd.com/) you can train the WMT'14 En2Fr model with
the following command:

```
$ DATA=... # path to the preprocessed dataset, must be visible from all nodes
$ PORT=9218 # any available tcp port that can be used by the trained to establish initial connection
$ sbatch --job-name fairseq-py --gres gpu:8 --nodes 16 --ntasks-per-node 8 \
    --cpus-per-task 10 --no-requeue --wrap 'srun --output train.log.node%t \
    --error train.stderr.node%t.%j python train.py $DATA --distributed-world-size 128 \
    --distributed-port $PORT --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
    --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
    --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --wd 0.0001'
```

Alternatively you'll need to manually start one process per each GPU:
```
$ DATA=... # path to the preprocessed dataset, must be visible from all nodes
$ HOST_PORT=your.devserver.com:9218 # has to be one of the hosts that will be used by the job \
    and the port on that host has to be available
$ RANK=... # the rank of this process, has to go from 0 to 127 in case of 128 GPUs
$ python train.py $DATA --distributed-world-size 128 \
      --force-anneal 50 --lr-scheduler fixed --max-epoch 55 \
      --arch fconv_wmt_en_fr --optimizer nag --lr 0.1,4 --max-tokens 3000 \
      --clip-norm 0.1 --dropout 0.1 --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.1 --wd 0.0001 \
      --distributed-init-method='tcp://$HOST_PORT' --distributed-rank=$RANK
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License
fairseq-py is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.
