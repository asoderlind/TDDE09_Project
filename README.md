# TDDE09_Project

## Data:

We set the seed `torch.manual_seed(12345)` for reproducibiilty.

### English treebank:

| Tagger   | Parsing system | Oracle  | Language | Accuracy | Unlabelled attachment score | Tags   |
| -------- | -------------- | ------- | -------- | -------- | --------------------------- | ------ |
| Standard | Arc-standard   | Static  | EN_EWT   | NA       | 0.7089                      | Golden |
| Standard | Arc-hybrid     | Static  | EN_EWT   | NA       | 0.6808                      | Golden |
| Standard | Arc-hybrid     | Dynamic | EN_EWT   | NA       | 0.6682                      | Golden |
| Standard | Arc-hybrid     | Static  | EN_EWT   | 0.8846   | 0.6604                      | Tagger |
| Standard | Arc-hybrid     | Static  | EN_EWT   | 0.8846   | 0.6219                      | Tagger |
| Standard | Arc-hybrid     | Dynamic | EN_EWT   | 0.8846   | 0.6239                      | Tagger |

### Japanese treebank:

| Tagger   | Parsing system | Oracle  | Language | Accuracy | Unlabelled attachment score | Tags   |
| -------- | -------------- | ------- | -------- | -------- | --------------------------- | ------ |
| Standard | Arc-standard   | Static  | JA_GSD   | 0.9548   | 0.8426                      | Golden |
| Standard | Arc-hybrid     | Static  | JA_GSD   | 0.9548   | 0.7628                      | Golden |
| Standard | Arc-hybrid     | Dynamic | JA_GSD   | 0.9548   | 0.7154                      | Golden |

## Structure

`baseline.py` - the main entrypoint

`parsing.py` - includes the parsing classes and `get_uas` evaluation function

`tagging.py` - includes the tagging classes and `accuracy` evaluation function

`models.py` - includes the base network `FixedWindowModel` that is trained

`utils.py` - includes any constants and helper functions such as `make_vocabs`

`treebank.py` - includes the `Treebank` class

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
