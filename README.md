# DMGL: Dynamic Multispace Geometric Learning for Multimodal Temporal Knowledge Graph Forecasting

This is the released codes of the anonymous submission to the AAAI 2026:

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
tqdm==4.66.2
numpy==1.26.4
```

## Introduction

- ``src``: Python scripts of the DMGL model.
- ``results``: Model files that replicate the reported results in our paper.
- ``data``: The four constructed MTKGs including ICE14-IMG-TXT, ICE18-IMG-TXT, ICE0515-IMG-TXT, GDELT-IMG-TXT.
- ``plms``: Pretrained models for encoding auxiliary modalities.
- ``preprocess``: Python scripts of data preprocessing. Specifically, `crawl_img.py` and `crawl_txt.py` collect time-sensitive images and texts for MTKGs. `1_picencode_dataset_vgg19.py` encodes images and applies mean pooling for each entity at different timestamps. `_fig_encode_matrices_dataset.py` stacks entity image embeddings to generate time-sensitive visual feature matrices. `docencode.py` encodes entity descriptions to produce time-sensitive linguistic feature matrices.
- ``pretrain``: Auxiliary linguistic and visual modality feature matrices of MTKGs.


## Training Command

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE14-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE0515-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE18-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset GDELT-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5
```

## Testing Command

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE14-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE0515-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset ICE18-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DMGL --dataset GDELT-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5 --test
```
