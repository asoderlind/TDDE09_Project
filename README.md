# TDDE09_Project

## Members:

Axel Söderlind ()
Erik Nordell ()
Linus Lundblad (linlu706)
Philip Welin-Berger ()

## Data:

We set the seed `torch.manual_seed(12345)` for reproducibiilty.

### English treebank:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.7089                      |
| arc-standard   | static  | 0.8846           | 0.6604                      |
| arc-hybrid     | static  | (Golden tags)    | 0.6699                      |
| arc-hybrid     | static  | 0.8846           | 0.6219                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.6682                      |
| arc-hybrid     | dynamic | 0.8846           | 0.6239                      |

With increased features:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.7097                      |
| arc-standard   | static  | 0.8870           | 0.6660                      |
| arc-hybrid     | static  | (Golden tags)    | 0.6735                      |
| arc-hybrid     | static  | 0.8870           | 0.6246                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.6665                      |
| arc-hybrid     | dynamic | 0.8870           | 0.6188                      |

Note that the non dynamic oracle runs take 3m and the dynamic oracle ones take 35m on CPU, GPU haven't been tested
Conclusion: The increased features choosen led to a worse result.

### Japanese treebank:

With increased features:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.8501                      |
| arc-standard   | static  | 0.9500           | 0.8314                      |
| arc-hybrid     | static  | (Golden tags)    | 0.7738                      |
| arc-hybrid     | static  | 0.9500           | 0.7587                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.7115                      |
| arc-hybrid     | dynamic | 0.9500           | 0.6947                      |

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
