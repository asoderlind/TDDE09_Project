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
| arc-standard   | static  | (Golden tags)    | 0.7379                      |
| arc-standard   | static  | 0.8870           | 0.6904                      | 
| arc-hybrid     | static  | (Golden tags)    | 0.7519                      |
| arc-hybrid     | static  | 0.8870           | 0.7001                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.7290                      |
| arc-hybrid     | dynamic | 0.8870           | 0.6804                      |

Note that the non dynamic oracle runs take 3m and the dynamic oracle ones take 58m on CPU, GPU haven't been tested

### Japanese treebank:

With increased features:

| Parsing system | Oracle  | Tagging Accuracy | Unlabelled attachment score |
| -------------- | ------- | ---------------- | --------------------------- |
| arc-standard   | static  | (Golden tags)    | 0.8601                      |
| arc-standard   | static  | 0.9521           | 0.8480                      |
| arc-hybrid     | static  | (Golden tags)    | 0.8686                      |
| arc-hybrid     | static  | 0.9521           | 0.8551                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.8112                      |
| arc-hybrid     | dynamic | 0.9521           | 0.7986                      |

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
