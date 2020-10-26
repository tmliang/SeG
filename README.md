# SeG
A pytorch implementation of the paper: [Self-Attention Enhanced Selective Gate with Entity-Aware Embedding for Distantly Supervised Relation Extraction](https://arxiv.org/pdf/1911.11899.pdf).

## Requirements
* python==3.8
* pytorch==1.6
* numpy==1.19.1
* tqdm==4.48.2
* scikit_learn==0.23.2

## Data
Download the dataset from [here](https://github.com/thunlp/HNRE/tree/master/raw_data), and unzip it under `./data/`.

## Train and Test
```
python main.py
```

## Experimental Result

|Model| P@100  | P@200 | P@300 | Mean | AUC |
| :- | :----: | :---: | :---: | :--: | :-: |
| SeG | 0.810 | 0.790 | 0.763 | 0.772 | 0.452 |
| PCNN+ATT+Ent | 0.810 | 0.790 | 0.763 | 0.772 | 0.452 |

## Note
PCNN and SAN **DO NOT** share the same entity-aware embedding layer, and the `lambda` values for PCNN and SAN are 0.05 and 1.0 respectively (confirmed by the authors).
