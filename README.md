# roboust-GAT

This project is the code for paper: RoGAT: A Robust GNN Combined Revised GAT with Adjusted Graphs



## Requirements:

```
dgl-cu100==0.4.3.post2
deeprobust==0
torch==1.5.0
networkx==2.4
numpy==1.19.2
```

or see the file of requirements.txt

## Usage

Run the file RoGAT/train.py   

or example:

```
python3 train.py --attack metattack --ptb_rate 0 --dataset cora --epoch 10 --lambda_ 1 
```

## Citation

```tex
@article{zhou2020rogat,
  title={RoGAT: a robust GNN combined revised GAT with adjusted graphs},
  author={Zhou, Xianchen and Wang, Hongxia},
  journal={arXiv preprint arXiv:2009.13038},
  year={2020}
}
```

