# TDDE09_Project

## Members:

Axel Söderlind (axeso712)

Erik Nordell (erino445)

Linus Lundblad (linlu706)

Philip Welin-Berger (phiwe030)

## Abstract:

In our project, we expanded the baseline with an arc-hybrid parser and dynamic oracle, augmenting the existing arc-standard parser and static oracle setup.

"Training Deterministic Parsers with Non-Deterministic Oracles" by Goldberg & Nivre indicates that the arc-hybrid parser should outperform the arc-standard parser, a result we confirmed with a 1% improvement in attachment score.[^1] However, performance varied across treebanks, with the Swedish treebank exhibiting lower scores.

While Goldberg & Nivre anticipated the dynamic oracle's superiority over the static oracle in arc-hybrid parsing, our project didn't align with this expectation.[^1] 
Despite efforts, our dynamic oracle consistently lagged behind the static one. We therefore decided on enhancing the dynamic oracle by implementing exploration parameters the same way "A Dynamic Oracle for Arc-Eager Dependency Parsing" by Goldberg & Nivre described.[^2] This yielded a 1% attachment score improvement. However, it still couldn't surpass the static oracle. This underscores the challenge of optimizing dynamic oracles. 

In conclusion our findings suggests that the arc-hybrid parser in combination with a static oracle produces the best attachment score out of the different methods we tried. 

[^1]: [Training Deterministic Parsers with Non-Deterministic Oracles](https://aclanthology.org/Q13-1033) (Goldberg & Nivre, TACL 2013)

[^2]: [A Dynamic Oracle for Arc-Eager Dependency Parsing](https://aclanthology.org/C12-1059) (Goldberg & Nivre, COLING 2012)

## Data:

We used batch_size = 15 with k = 2 and p = 0.9 as our exploration parameters for the dynamic oracle. We choose these specific parameters as we found that they produced the best results.
The result shows the average values over 5 different seeds. The seeds used were 1, 2, 3, 4 and 5 and 
the same seed was both used for random.seed(SEED) and torch.manual_seed(SEED) for reproducibility.

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

| Parsing system | Oracle  | Avarage Tagging Accuracy | Avarage Unlabelled attachment score |
| -------------- | ------- | ------------------------ | ----------------------------------- |
| arc-standard   | static  | (Golden tags)            | 85.67%                              |
| arc-standard   | static  | 94.992%                  | 83.99%                              |
| arc-hybrid     | static  | (Golden tags)            | 86.75%                              |
| arc-hybrid     | static  | 94.992%                  | 85.06%                              |
| arc-hybrid     | dynamic | (Golden tags)            | 79.07%                              |
| arc-hybrid     | dynamic | 94.992%                  | 77.50%                              |

### Swedish treebank:

| Parsing system | Oracle  | Avarage Tagging Accuracy | Avarage Unlabelled attachment score |
| -------------- | ------- | ------------------------ | ----------------------------------- |
| arc-standard   | static  | (Golden tags)            | 71.00%                              |
| arc-standard   | static  | 90.276%                  | 63.53%                              |
| arc-hybrid     | static  | (Golden tags)            | 69.73%                              |
| arc-hybrid     | static  | 90.276%                  | 62.42%                              |
| arc-hybrid     | dynamic | (Golden tags)            | 68.82%                              |
| arc-hybrid     | dynamic | 90.276%                  | 61.69%                              |

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
