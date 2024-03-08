# TDDE09_Project

## Members:

Axel Söderlind (axeso712)

Erik Nordell (erino445)

Linus Lundblad (linlu706)

Philip Welin-Berger (phiwe030)

## Abstract:

In this project we extended the baseline, which included an arc-standard parser with a static oracle,
with an arc-hybrid parser and a dynamic oracle. The dynamic oracle may only be used with the arc-hybrid parser,
however, since it requires a parsing system that is arc-decomposable. According to the litterature, the arc-hybrid
parser with a static oracle should perform equally well as the arc-standard. It was also found that the dynamic oracle
should perform better than the static oracle, both using arc-hybrid parsing.[^1] 
To improve the dynamic oracle further we choose to implement exploration parametres for the dynamic oracle according to "A Dynamic Oracle for Arc-Eager Dependency Parsin" 
by Goldberg & Nivre. Our result indicates... [^2]

[^1]: [Training Deterministic Parsers with Non-Deterministic Oracles](https://aclanthology.org/Q13-1033) (Goldberg & Nivre, TACL 2013)

[^2]: [A Dynamic Oracle for Arc-Eager Dependency Parsing](https://aclanthology.org/C12-1059) (Goldberg & Nivre, COLING 2012)

## Data:

We used batch_size = 15 with k = 2 and p = 0.9 as our exploration parametres for the dynamic oracle.
The result shows the avarage values over 5 different seeds. The seeds used were 1, 2, 3, 4 and 5 and 
the same seed were both used for random.seed(SEED) and torch.manual_seed(SEED) for reproducibility.

### English treebank:

| Parsing system | Oracle  | Avarage Tagging Accuracy | Avarage Unlabelled attachment score |
| -------------- | ------- | ------------------------ | ----------------------------------- |
| arc-standard   | static  | (Golden tags)            | 73.58%                              |
| arc-standard   | static  | 88.084%                  | 68.71%                              |
| arc-hybrid     | static  | (Golden tags)            | 74.54%                              |
| arc-hybrid     | static  | 88.084%                  | 69.50%                              |
| arc-hybrid     | dynamic | (Golden tags)            | 73.60%                              |
| arc-hybrid     | dynamic | 88.084%                  | 68.70%                              |

### Japanese treebank:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.8496                      |
| arc-standard   | static  | 0.9511           | 0.8389                      |
| arc-hybrid     | static  | (Golden tags)    | 0.8710                      |
| arc-hybrid     | static  | 0.9511           | 0.8538                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.7797                      |
| arc-hybrid     | dynamic | 0.9511           | 0.7672                      |

### Swedish treebank:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.7145                      |
| arc-standard   | static  | 0.9035           | 0.6436                      |
| arc-hybrid     | static  | (Golden tags)    | 0.6774                      |
| arc-hybrid     | static  | 0.9035           | 0.6179                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.6883                      |
| arc-hybrid     | dynamic | 0.9035           | 0.6179                      |

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

en_ewt, ja_gsd, sv_talbanken
