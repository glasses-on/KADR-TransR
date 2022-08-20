# KGCN, KGAT with TransR
This is PyTorch implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.7.10. The required packages are as follows:
* torch == 1.6.0
* numpy == 1.21.4
* pandas == 1.3.5
* scipy == 1.5.2
* tqdm == 4.62.3
* scikit-learn == 1.0.1

## Run the Codes

* KGAT
```
python main_kgat.py --data_name 'amazon-book' --exp_name "your_exp_name" --use_pretrain 0```
```
* KGCN
   * To Train with TransR:
```
python main_kgcn.py --data_name 'last_fm_new' --exp_name "your_exp_name" --train_kg 1  --use_pretrain 0```
```
* KGCN
   * To Train without TransR - Vanilla KGCN:
```
python main_kgcn.py --data_name 'last_fm_new' --exp_name "your_exp_name" --train_kg 0  --use_pretrain 0```
```

## Related Papers

* KGAT
    * Proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD 2019.
    * Implementation by the paper authors: [https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.
