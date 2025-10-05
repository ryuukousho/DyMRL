# DyMRL: Dynamic Multispace Representation Learning for Multimodal Event Forecasting in Knowledge Graph

This is the released codes of the anonymous submission to the WWW 2026:

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
- ``preprocess``: Python scripts of data preprocessing. Specifically, `crawl_img.py` and `crawl_txt.py` collect time-sensitive images and texts for multimodal temporal KGs. `1_picencode_dataset_vgg19.py` encodes images and applies mean pooling for each event (entity) at different timestamps. `_fig_encode_matrices_dataset.py` stacks image embeddings to generate time-sensitive visual feature matrices. `docencode.py` encodes descriptions to produce time-sensitive linguistic feature matrices.
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
