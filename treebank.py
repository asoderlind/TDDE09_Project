import conllu
from torch.utils.data import Dataset

from projectivize import filename_projectivize


class Treebank(Dataset):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.items: list[list[tuple[str, str, int]]] = []
        with open(filename, encoding="utf-8") as fp:
            for tokens in conllu.parse_incr(fp):
                sentence: list[tuple[str, str, int]] = [("[ROOT]", "[ROOT]", 0)]
                for token in tokens.filter(id=lambda x: isinstance(x, int)):
                    sentence.append((token["form"], token["upos"], token["head"]))
                self.items.append(sentence)

    @classmethod
    def from_non_projective(cls, filename: str) -> "Treebank":
        filename_projectivize(filename, "tmp")
        return cls("tmp")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
