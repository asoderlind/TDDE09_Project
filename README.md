# TDDE09_Project

## Data:

We set the seed `torch.manual_seed(0)` for reproducibiilty.

| Tagger   | Parsing system | Oracle  | Language | Accuracy | Unlabelled attachment score |
| -------- | -------------- | ------- | -------- | -------- | --------------------------- |
| Standard | Arc-standard   | Static  | EN_EWT   | 0.8827   | 0.6619                      |
| Standard | Arc-hybrid     | Static  | EN_EWT   | 0.8827   | 0.6244                      |
| Standard | Arc-standard   | Static  | JA_GSD   | 0.9548   | 0.8426                      |
| Standard | Arc-hybrid     | Static  | JA_GSD   | 0.9548   | 0.7628                      |
| Standard | Arc-hybrid     | Dynamic | JA_GSD   | 0.9548   | 0.7154                      |

## Usage

### Running on all treebanks

```shell
python3 baseline.py
```

### Running on specific treebank

```shell
python3 baseline.py {treebank}
```

### Currently the avaiable treebanks are

en_ewt, ja_gsd
