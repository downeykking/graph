## Introdcution
In this repo, we implement some common graph convolutional neural network layers (GCN, GAT, GraphSAGE...)

And in `layers`, we have two branchs: `pyg` and `pytorch`.
Both ways implemented convolutional layers. 


## Usage

```
git clone https://github.com/downeykking/graph.git
cd directory gat or gcn
run main.py
```



## Performances：

| model         | GCN     | GAT       |
| :------------ | ------- | --------- |
| epoch         | 200     | 5000      |
| learning rate | 0.01    | 0.005     |
| dropout       | 0.5     | 0.6       |
| weight decay  | 5e-4    | 5e-4      |
| hidden        | 16      | 8         |
| seed          | 2022    | 2022      |
| head          | /       | 8         |
| alpha         | /       | 0.2       |
| epoch time    | 0.0031s | 0.0198s   |
| total time    | 0.6171s | 199.6522s |
| loss          | 0.4107  | 0.5054    |
| test acc      | 82.60   | 84.40     |



## Requirements：

```
torch=1.10.0+cu102
torch-geometric=2.0.2
pandas=1.3.4
numpy=1.21.4
```

