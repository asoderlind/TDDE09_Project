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
| arc-standard   | static  | (Golden tags)    | 0.7419                      |
| arc-standard   | static  | 0.8846           | 0.6945                      |
| arc-hybrid     | static  | (Golden tags)    | 0.7535                      |
| arc-hybrid     | static  | 0.8846           | 0.7032                      |
| arc-hybrid     | dynamic | (Golden tags)    | 0.7301                      |
| arc-hybrid     | dynamic | 0.8846           | 0.6815                      |

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
