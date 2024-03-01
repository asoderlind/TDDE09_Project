# TDDE09_Project

## Data:

For the Arc-standard + Static oracle implementation:

| Tagger   | Parsing system | Oracle | Language | Accuracy | Unlabelled attachment score |
| -------- | -------------- | ------ | -------- | -------- | --------------------------- |
| Standard | Arc-standard   | Static | EN_EWT   | 0.8804   | 0.6612                      |
| Standard | Arc-standard   | Static | JA_GSD   | 0.9486   | 0.8389                      |

For the Arc-hybrid + Static oracle implementation:

| Tagger   | Parsing system | Oracle | Language | Accuracy | Unlabelled attachment score |
| -------- | -------------- | ------ | -------- | -------- | --------------------------- |
| Standard | Arc-hybrid     | Static | EN_EWT   | TBA      | TBA                         |
| Standard | Arc-hybrid     | Static | JA_GSD   | TBA      | TBA                         |

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
