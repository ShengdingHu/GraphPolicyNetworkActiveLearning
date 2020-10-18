# Graph Policy Network for Transferable Active Learning on Graphs
This is the code of the paper **G**raph **P**olicy network for transferable **A**ctive learning on graphs (GPA). 

## Dependencies
matplotlib==2.2.3
networkx==2.4
scikit-learn==0.21.2
numpy==1.16.3
scipy==1.2.1
torch==1.3.1

## Data
We have provided Cora, Pubmed, Citeseer, Reddit1401 whose data format have been processed and can be directly consumed by our code. Reddit1401 is collected from the reddit data source (where 1401 means Janurary, 2014) and preprocessed by ourselves. If you use these graphs in your work, please cite our paper. For the Coauthor_CS and Coauthor_Phy dataset, we don't provide the processed data because they are too large for github repos. If you are interested, please email shengdinghu@gmail.com  for the processed data.

## Train

Use ```train.py``` to train the active learning policy on multiple labeled training graphs. Assume that we have two labeled training graphs ```A``` and ```B``` with query budgets of ```x``` and ```y``` respectively, and we want to save the trained model in ```temp.pkl```, then use the following commend: 
```
python -m src.train --datasets A+B --budgets x+y  --save 1 --savename temp
```
Please refer to the source code to see how to set the other arguments. 

## Test
Use ```test.py``` to test the learned active learning policy on unlabeled test graphs. Assume that we have an unlabeled test graph ```G``` with a query budget of ```z```, and we want to test the policy stored in ```temp.pkl```, then use the following commend: 
```
python -m src.test --method 3 --modelname temp --datasets G --budgets z
```
Please refer to the source code to see how to set the other arguments. 

## Pre-trained Models and Results
We provide several pre-trained models with their test results on the unlabeled test graphs. 
For transferable active learning on graphs from the **same** domain, we train on Reddit {1, 2} on test on Reddit {3, 4, 5}. The pre-trained model is saved in ```models/pretrain_reddit1+2.pkl```. The test results are

| Metric | Reddit 3 | Reddit 4 | Reddit 5 |
| :---: | :---:| :---:| :---: |
| Micro-F1 | 92.51 | 91.49 | 90.71 |
| Macro-F1 | 92.22 | 89.57 | 90.32 |

For tranferable active learning on graphs across **different** domains, we provide three pre-trained models trained on different training graphs as follows: 

1. Train on Cora + Citeseer, and test on the remaining graphs. The pre-trained model is saved in ```models/pretrain_cora+citeseer.pkl```. The test results are

| Metric | Pubmed | Reddit 1 | Reddit 2 | Reddit 3 | Reddit 4 | Reddit 5 | Physics | CS |
| :---: | :---:| :---:| :---: | :---: | :---:| :---:| :---: | :---: |
| Micro-F1 | 77.44 | 88.16 | 95.25 | 92.09 | 91.37 | 90.71 | 87.91 | 87.64 |
| Macro-F1 | 75.28 | 87.84 | 95.04 | 91.77 | 89.50 | 90.30 | 82.57 | 84.45 |

2. Train on Cora + Pubmed, and test on the remaining graphs. The pre-trained model is saved in ```models/pretrain_cora+pubmed.pkl```. The test results are

| Metric | Citeseer | Reddit 1 | Reddit 2 | Reddit 3 | Reddit 4 | Reddit 5 | Physics | CS |
| :---: | :---:| :---:| :---: | :---: | :---:| :---:| :---: | :---: |
| Micro-F1 | 65.76 | 88.14 | 95.14 | 92.08 | 91.05 | 90.38 | 87.14 | 88.15 |
| Macro-F1 | 57.52 | 87.86 | 94.93 | 91.78 | 89.08 | 89.92 | 81.04 | 85.24 |

3. Train on Citeseer + Pubmed, and test on the remaining graphs. The pre-trained model is saved in ```models/pretrain_citeseer+pubmed.pkl```. The test results are

| Metric | Cora | Reddit 1 | Reddit 2 | Reddit 3 | Reddit 4 | Reddit 5 | Physics | CS |
| :---: | :---:| :---:| :---: | :---: | :---:| :---:| :---: | :---: |
| Micro-F1 | 73.40 | 87.57 | 95.08 | 92.07 | 90.99 | 90.53 | 87.06 | 87.00 |
| Macro-F1 | 71.22 | 87.11 | 94.87 | 91.74 | 88.97 | 90.14 | 81.20 | 83.90 |
