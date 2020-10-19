# SeG
A pytorch implementation of the paper: [Self-Attention Enhanced Selective Gate with Entity-Aware Embedding for Distantly Supervised Relation Extraction](https://arxiv.org/pdf/1911.11899.pdf).

## Requirements
* python==3.8
* pytorch==1.6
* numpy==1.19.1
* tqdm==4.48.2
* scikit_learn==0.23.2

## Data
Download the dataset from [here](https://github.com/thunlp/HNRE/tree/master/raw_data), and unzip under './data/'.

## Train and Test
```
python main.py
```

## Experimental Result

| AUC | P@100  | P@200 | P@300 | Mean |
| :----: | :----: | :----: | :----: | :----: |
| --- | ------ | ----- | ----- | ---- |
