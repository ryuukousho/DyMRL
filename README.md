# DyMRL: Dynamic Multispace Representation Learning for Multimodal Event Forecasting in Knowledge Graph

This is the released codes of the work in the ACM Web Conference 2026 (WWW'26):

## Environment

```shell
python==3.10.9
torch==2.2.1+cu118
dgl==2.1.0+cu118
tqdm==4.66.2
numpy==1.26.4
```

## Introduction

- ``src``: Python scripts of the DyMRL model.
- ``results``: Model files that replicate the reported results in our paper.
- ``data``: The four constructed multimodal temporal KGs including ICE14-IMG-TXT, ICE18-IMG-TXT, ICE0515-IMG-TXT, GDELT-IMG-TXT.
- ``plms``: Pretrained models for encoding auxiliary modalities.
- ``pretrain``: Auxiliary linguistic and visual modality feature matrices of multimodal temporal KGs.


## Training Command

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE14-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE0515-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE18-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3
```

```shell
cd src
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset GDELT-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5
```

## Testing Command

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE14-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 6 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE0515-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset ICE18-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 3 --test
```

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model DyMRL --dataset GDELT-IMG-TXT --bias learn --s-delta-ind --n-head 2 --rank 20 --history-len 5 --test
```
