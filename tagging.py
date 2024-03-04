from collections.abc import Iterable
from typing import cast

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from models import FixedWindowModel
from treebank import Treebank
from utils import PAD_IDX, UNK_IDX, make_vocabs


class Tagger:
    def predict(self, sentence):
        raise NotImplementedError


class FixedWindowTagger(Tagger):
    def __init__(
        self,
        vocab_words: dict[str, int],
        vocab_tags: dict[str, int],
        word_dim: int = 50,
        tag_dim: int = 10,
        hidden_dim: int = 100,
    ) -> None:
        embedding_specs: list[tuple[int, int, int]] = [
            (3, len(vocab_words), word_dim),
            (1, len(vocab_tags), tag_dim),
        ]
        self.model = FixedWindowModel(embedding_specs, hidden_dim, len(vocab_tags))
        self.w2i = vocab_words
        self.i2t = {i: t for t, i in vocab_tags.items()}

    def featurize(self, words: list[int], i: int, pred_tags: list[int]) -> torch.Tensor:
        x = torch.zeros(4, dtype=torch.long)
        x[0] = words[i]
        x[1] = words[i - 1] if i > 0 else PAD_IDX
        x[2] = words[i + 1] if i + 1 < len(words) else PAD_IDX
        x[3] = pred_tags[i - 1] if i > 0 else PAD_IDX
        return x

    def predict(self, words: list[str]) -> list[str]:
        words: list[int] = [self.w2i.get(w, UNK_IDX) for w in words]
        pred_tags: list[int] = []
        for i in range(len(words)):
            features = self.featurize(words, i, pred_tags)
            with torch.no_grad():
                scores = self.model.forward(features)
            pred_tag: int = scores.argmax().item()
            pred_tags.append(pred_tag)
        return [self.i2t[i] for i in pred_tags]


def training_examples_tagger(
    vocab_words: dict[str, int],
    vocab_tags: dict[str, int],
    gold_data: Treebank,
    tagger: FixedWindowTagger,
    batch_size: int = 100,
    shuffle: bool = False,
) -> Iterable[tuple[torch.Tensor, torch.LongTensor]]:
    bx = []
    by = []
    for sentence in gold_data:
        # Separate the words and the gold-standard tags
        words, gold_tags, _ = zip(*sentence)

        # Encode words and tags using the vocabularies
        words = [vocab_words.get(w, UNK_IDX) for w in words]
        gold_tags = [vocab_tags[t] for t in gold_tags]

        # Simulate a run of the tagger over the sentence, collecting training examples
        pred_tags = []
        for i, gold_tag in enumerate(gold_tags):
            bx.append(tagger.featurize(words, i, pred_tags))
            by.append(gold_tag)
            if len(bx) >= batch_size:
                bx = torch.stack(bx)
                by = torch.LongTensor(by)
                if shuffle:
                    random_indices = torch.randperm(len(bx))
                    yield bx[random_indices], by[random_indices]
                else:
                    yield bx, by
                bx = []
                by = []
            pred_tags.append(gold_tag)  # teacher forcing!

    # Check whether there is an incomplete batch
    if bx:
        bx = torch.stack(bx)
        by = torch.LongTensor(by)
        if shuffle:
            random_indices = torch.randperm(len(bx))
            yield bx[random_indices], by[random_indices]
        else:
            yield bx, by


def train_tagger(
    train_data: Treebank,
    n_epochs: int = 1,
    batch_size: int = 100,  # noqa: ARG001
    lr: float = 1e-2,
) -> FixedWindowTagger:
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)

    # Instantiate the tagger
    tagger = FixedWindowTagger(vocab_words, vocab_tags)

    # Instantiate the optimizer
    optimizer = optim.Adam(tagger.model.parameters(), lr=lr)

    # Training loop
    for _ in range(n_epochs):
        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(len(s) for s in train_data)) as pbar:
            for bx, by in training_examples_tagger(
                vocab_words, vocab_tags, train_data, tagger
            ):
                optimizer.zero_grad()
                output = tagger.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss / n_examples)
                pbar.update(len(bx))

    return tagger


def accuracy(tagger: FixedWindowTagger, gold_data: Treebank) -> float:
    correct = 0
    total = 0
    for sentence in gold_data:
        sentence = cast(list[tuple[str, str, int]], sentence)
        zipped_object: tuple[list[str], list[str], list[int]] = zip(*sentence)
        words, gold_tags, _ = zipped_object
        pred_tags = tagger.predict(words)
        for gold_tag, pred_tag in zip(
            gold_tags[1:], pred_tags[1:]
        ):  # ignore the pseudo-root
            correct += int(gold_tag == pred_tag)
            total += 1
    return correct / total
