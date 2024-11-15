# Adapted Use

This repository was adapted for validating LeaPformers. We thank the team behind Skyformers for their work on this PyTorch-based LRA implementation and we ask that anyone who uses this adapted version of the repo consider citing their work as well. Example scripts to get LeaPformers working here can be found in `src/scripts` and should require only minor modifications.

## Requirements

To install requirements in a Python virtual environment:
```
python3 -m venv pytorch_lra_venv
pip install -r requirements.txt
```

Note: Specific requirements for data preprocessing are not included here. A conda-based approach would also work perfectly well. 


## Data Preparation

Processed files can be downloaded [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing), or processed with the following steps:

1. Requirements
```
tensorboard>=2.3.0
tensorflow>=2.3.1
tensorflow-datasets>=4.0.1
```
2. Download [the TFDS files for pathfinder](https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz) and then set _PATHFINER_TFDS_PATH to the unzipped directory (following https://github.com/google-research/long-range-arena/issues/11)
3. Download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) (7.7 GB).
4. Unzip `lra-release` and put under `./data/`.
```
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar zxvf lra-release.gz 
```
5. Create a directory `lra_processed` under `./data/`.
```
mkdir lra_processed
cd ..
```
6.The directory structure would be (assuming the root dir is `code`)
```
./data/lra-processed
./data/long-range-arena-main
./data/lra_release
```
7. Create train, dev, and test dataset pickle files for each task.
```
cd preprocess
python create_pathfinder.py
python create_listops.py
python create_retrieval.py
python create_text.py
python create_cifar10.py
```

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena).



## Run 

Modify the configuration in `config.py` and run
```
python main.py --mode train --attn skyformer --task lra-text
```
- mode: `train`, `eval` (`eval` currently bugged, fix is TODO)
- attn: `softmax`, `nystrom`, `linformer`, `reformer`, `perfromer`, `informer`, `bigbird`,  `kernelized`, `skyformer`, ... (see [src/models](src/models) for more)
- task: `lra-listops`, `lra-pathfinder`, `lra-retrieval`, `lra-text`, `lra-image`


## Reference

If you use this LRA implementation to test out efficient attention mechanisms, you should cite Skyformer. The changes made to enable their implementation were extremely superficial. We mostly added some extra attention mechanisms and that's about it.

```bibtex
@inproceedings{Skyformer,
    title={Skyformer: Remodel Self-Attention with Gaussian Kernel and Nystr\"om Method}, 
    author={Yifan Chen and 
            Qi Zeng and 
            Heng Ji and 
            Yun Yang},
    booktitle={Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December
               6-14, 2021, virtual},
    year={2021}
}

```
