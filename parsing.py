from collections.abc import Iterable
from typing import cast

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from models import FixedWindowModel
from treebank import Treebank
from utils import PAD_IDX, UNK_IDX, make_vocabs


class Parser:
    def predict(self, words, tags):
        raise NotImplementedError


# --------------------------
# ARC STANDARD IMPLEMENTATION
# --------------------------
class ArcStandardParser(Parser):
    MOVES = tuple(range(3))

    SH, LA, RA = MOVES

    @staticmethod
    def initial_config(num_words: int) -> tuple[int, list, list[int]]:
        return 0, [], [0] * num_words

    @staticmethod
    def valid_moves(config: tuple[int, list, list[int]]) -> list[int]:
        pos, stack, heads = config
        moves: list[int] = []
        if pos < len(heads):
            moves.append(ArcStandardParser.SH)
        if len(stack) >= 3:  # disallow LA with root as dependent
            moves.append(ArcStandardParser.LA)
        if len(stack) >= 2:
            moves.append(ArcStandardParser.RA)
        return moves

    @staticmethod
    def next_config(
        config: tuple[int, list, list[int]], move: int
    ) -> tuple[int, list, list[int]]:
        pos, stack, heads = config
        stack = list(stack)  # copy because we will modify it
        if move == ArcStandardParser.SH:
            stack.append(pos)
            pos += 1
        else:
            heads = list(heads)  # copy because we will modify it
            s1 = stack.pop()
            s2 = stack.pop()
            if move == ArcStandardParser.LA:
                heads[s2] = s1
                stack.append(s1)
            if move == ArcStandardParser.RA:
                heads[s1] = s2
                stack.append(s2)
        return pos, stack, heads

    @staticmethod
    def is_final_config(config: tuple[int, list, list[int]]) -> bool:
        pos, stack, heads = config
        return pos == len(heads) and len(stack) == 1


# --------------------------
# ARC HYBRID IMPLEMENTATION
#   - LA from arc-eager
#   - rest from arc-standard
# --------------------------
class ArcHybridParser(Parser):
    MOVES = tuple(range(3))

    SH, LA, RA = MOVES

    @staticmethod
    def initial_config(num_words: int) -> tuple[int, list, list[int]]:
        return 0, [], [0] * num_words

    @staticmethod
    def valid_moves(config: tuple[int, list, list[int]]) -> list[int]:
        pos, stack, heads = config
        moves: list[int] = []
        if pos < len(heads):
            moves.append(ArcHybridParser.SH)
        if len(stack) >= 1 and pos < len(heads):
            moves.append(ArcHybridParser.LA)
        if len(stack) >= 2:
            moves.append(ArcHybridParser.RA)
        return moves

    @staticmethod
    def next_config(
        config: tuple[int, list[int], list[int]], move: int
    ) -> tuple[int, list[int], list[int]]:
        pos, stack, heads = config
        if move == ArcHybridParser.SH:
            stack.append(pos)
            pos += 1
        elif move == ArcHybridParser.LA:
            s1 = stack.pop()
            heads[s1] = pos  # Arc from front of buffer to top of stack
        elif move == ArcHybridParser.RA:  # arc-standard
            s1 = stack.pop()
            s2 = stack[-1]
            heads[s1] = s2
        return pos, stack, heads

    @staticmethod
    def is_final_config(config: tuple[int, list, list[int]]) -> bool:
        pos, stack, heads = config
        return pos == len(heads) and len(stack) == 1 and stack[0] == 0

    # True if it can be done at zero cost
    @staticmethod
    def zero_cost_sh(
        config: tuple[int, list[int], list[int]], gold_heads: list[int]
    ) -> bool:
        pos, stack, heads = config

        # avoid out of range error
        if len(stack) == 0:
            return True

        # check so that the position that's about to be shifted isn't a gold head for a
        # position in the stack or the reverse (in that case LA/RA might be zero cost)
        for stack_pos in stack[0:-1]:
            if gold_heads[stack_pos] == pos:
                return False
            if gold_heads[pos] == stack_pos:
                return False

        return True

    @staticmethod
    def zero_cost_la(
        config: tuple[int, list[int], list[int]], gold_heads: list[int]
    ) -> bool:
        pos, stack, heads = config

        # if we pop stack[-1], it won't be able to find its gold-standard dependant.
        for buffer_pos in range(pos, len(heads)):
            if stack[-1] == gold_heads[buffer_pos]:
                return False

        # if we pop stack[-1], it won't be able to find its gold-standard head.
        if len(stack) > 1:
            if stack[-2] == gold_heads[stack[-1]]:
                return False

        # if we pop stack[-1], it won't be able to find its gold-standard head.
        for buffer_pos in range(pos + 1, len(heads)):
            if buffer_pos == gold_heads[stack[-1]]:
                return False
        return True

    @staticmethod
    def zero_cost_ra(
        config: tuple[int, list[int], list[int]], gold_heads: list[int]
    ) -> bool:
        pos, stack, heads = config

        # if we pop stack[-1], it won't be able to find its gold-standard dependant.
        for buffer_pos in range(pos, len(heads)):
            if stack[-1] == gold_heads[buffer_pos]:
                return False

        # making a RA transition will pop stack[-1], it won't be able to find its
        # gold-standard head.
        for buffer_pos in range(pos, len(heads)):
            if buffer_pos == gold_heads[stack[-1]]:
                return False
        return True


class FixedWindowParserBase:
    def __init__(
        self,
        vocab_words: dict[str, int],
        vocab_tags: dict[str, int],
        word_dim: int = 50,
        tag_dim: int = 10,
        hidden_dim: int = 180,
    ):
        embedding_specs: list[tuple[int, int, int]] = [
            (3, len(vocab_words), word_dim),
            (3, len(vocab_tags), tag_dim),
        ]
        self.model = FixedWindowModel(
            embedding_specs, hidden_dim, len(ArcStandardParser.MOVES)
        )
        self.w2i = vocab_words
        self.t2i = vocab_tags

    def featurize(
        self, words: list[int], tags: list[int], config: tuple[int, list, list[int]]
    ) -> torch.Tensor:
        i, stack, heads = config
        x = torch.zeros(14, dtype=torch.long)
        # word at current pos
        x[0] = words[i] if i < len(words) else PAD_IDX
        # first stack
        x[1] = words[stack[-1]] if len(stack) >= 1 else PAD_IDX
        # second stack
        x[2] = words[stack[-2]] if len(stack) >= 2 else PAD_IDX
        x[3] = tags[i] if i < len(tags) else PAD_IDX
        x[4] = tags[stack[-1]] if len(stack) >= 1 else PAD_IDX
        x[5] = tags[stack[-2]] if len(stack) >= 2 else PAD_IDX
        # our expanded tags:
        # the following does slightly improve both tagging accuracy and UAS
        x[6] = words[i + 1] if i + 1 < len(words) else PAD_IDX
        x[7] = tags[i + 1] if i + 1 < len(tags) else PAD_IDX
        x[8] = words[i + 2] if i + 2 < len(words) else PAD_IDX
        x[9] = tags[i + 2] if i + 2 < len(tags) else PAD_IDX
        x[10] = words[stack[-3]] if len(stack) >= 3 else PAD_IDX
        x[11] = tags[stack[-3]] if len(stack) >= 3 else PAD_IDX

        # adding features like this one won't make a difference since it will be embedded
        # adding bigrams for the words and tags doesn't seem to improve the result at all
        # x[12] = (
        #    (words[i] * len(self.w2i)) + words[i + 1] if i + 1 < len(words) else PAD_IDX
        # )
        # x[13] = (
        #    (tags[i] * len(self.t2i)) + tags[i + 1] if i + 1 < len(tags) else PAD_IDX
        # )

        return x

    def predict(self, words: list[str], tags: list[str]) -> list[int]:
        words: list[int] = [self.w2i.get(w, UNK_IDX) for w in words]
        tags: list[int] = [self.t2i.get(t, UNK_IDX) for t in tags]
        config = self.initial_config(len(words))
        valid_moves = self.valid_moves(config)
        while valid_moves:
            features = self.featurize(words, tags, config)
            with torch.no_grad():
                scores = self.model.forward(features)

            # We may only predict valid transitions
            best_score, pred_move = float("-inf"), None
            for move in valid_moves:
                if scores[move] > best_score:
                    best_score, pred_move = scores[move], move

            config = self.next_config(config, pred_move)
            valid_moves = self.valid_moves(config)
        config = cast(tuple[int, list, list[int]], config)
        i, stack, pred_heads = config
        return pred_heads


class FixedWindowParser(FixedWindowParserBase, ArcStandardParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FixedWindowParserHybrid(FixedWindowParserBase, ArcHybridParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def oracle_moves(
    gold_heads: Treebank,
) -> Iterable[tuple[tuple[int, list, list[int]], int]]:
    # Keep track of how many dependents each head still needs to find
    remaining_count = [0] * len(gold_heads)
    for node in gold_heads:
        remaining_count[node] += 1

    # Simulate a parser
    config = ArcStandardParser.initial_config(len(gold_heads))
    while not ArcStandardParser.is_final_config(config):
        pos, stack, heads = config
        if len(stack) >= 2:
            s1 = stack[-1]
            s2 = stack[-2]
            if gold_heads[s2] == s1 and remaining_count[s2] == 0:
                move = ArcStandardParser.LA
                yield config, move
                config = ArcStandardParser.next_config(config, move)
                remaining_count[s1] -= 1
                continue
            if gold_heads[s1] == s2 and remaining_count[s1] == 0:
                move = ArcStandardParser.RA
                yield config, move
                config = ArcStandardParser.next_config(config, move)
                remaining_count[s2] -= 1
                continue
        move = ArcStandardParser.SH
        yield config, move
        config = ArcStandardParser.next_config(config, move)


def oracle_moves_hybrid(
    gold_heads: Treebank,
) -> Iterable[tuple[tuple[int, list, list[int]], int]]:
    # Keep track of how many dependents each head still needs to find
    remaining_count = [0] * len(gold_heads)
    for node in gold_heads:
        remaining_count[node] += 1

    # Simulate a parser
    config = ArcHybridParser.initial_config(len(gold_heads))
    while not ArcHybridParser.is_final_config(config):
        pos, stack, heads = config
        if len(stack) >= 1:
            s1 = stack[-1]
            if gold_heads[s1] == pos and remaining_count[s1] == 0:
                move = ArcHybridParser.LA
                yield config, move
                config = ArcHybridParser.next_config(config, move)
                remaining_count[pos] -= 1
                continue
        if len(stack) >= 2:
            s1 = stack[-1]
            s2 = stack[-2]
            if gold_heads[s1] == s2 and remaining_count[s1] == 0:
                move = ArcHybridParser.RA
                yield config, move
                config = ArcHybridParser.next_config(config, move)
                remaining_count[s2] -= 1
                continue
        move = ArcHybridParser.SH
        yield config, move
        config = ArcHybridParser.next_config(config, move)


def training_examples_parser(
    vocab_words: dict[str, int],
    vocab_tags: dict[str, int],
    gold_data: Treebank,
    parser: FixedWindowParser | FixedWindowParserHybrid,
    batch_size: int = 100,
) -> Iterable[tuple[torch.Tensor, torch.LongTensor]]:
    bx = []
    by = []

    is_hybrid = isinstance(parser, FixedWindowParserHybrid)

    for sentence in gold_data:
        # Separate the words, gold tags, and gold heads
        words, tags, gold_heads = zip(*sentence)

        oracle_moves_fn = oracle_moves_hybrid if is_hybrid else oracle_moves

        # Encode words and tags using the vocabularies
        words = [vocab_words.get(w, UNK_IDX) for w in words]
        tags = [vocab_tags[t] for t in tags]

        # Call the oracle
        for config, gold_move in oracle_moves_fn(gold_heads):
            bx.append(parser.featurize(words, tags, config))
            by.append(gold_move)
            if len(bx) >= batch_size:
                bx = torch.stack(bx)
                by = torch.LongTensor(by)
                yield bx, by
                bx = []
                by = []

    # Check whether there is an incomplete batch
    if bx:
        bx = torch.stack(bx)
        by = torch.LongTensor(by)
        yield bx, by


def train_parser(
    train_data: Treebank,
    parser_type: str = "arc-standard",
    n_epochs: int = 1,
    batch_size: int = 100,  # noqa: ARG001
    lr: float = 1e-2,
) -> FixedWindowParserHybrid:
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)

    # Instantiate the parser
    parser = (
        FixedWindowParserHybrid(vocab_words, vocab_tags)
        if parser_type == "arc-hybrid"
        else FixedWindowParser(vocab_words, vocab_tags)
    )

    # Instantiate the optimizer
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    # Training loop
    for _ in range(n_epochs):
        running_loss = 0
        n_examples = 0
        with tqdm(total=sum(2 * len(s) - 1 for s in train_data)) as pbar:
            for bx, by in training_examples_parser(
                vocab_words, vocab_tags, train_data, parser
            ):
                optimizer.zero_grad()
                output = parser.model.forward(bx)
                loss = F.cross_entropy(output, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_examples += 1
                pbar.set_postfix(loss=running_loss / n_examples)
                pbar.update(len(bx))

    return parser


def dynamic_oracle(
    config: tuple[int, list[int], list[int]],
    valid_moves: list[int],
    gold_heads: list[int],
) -> list[int]:
    moves: list[int] = []
    if ArcHybridParser.SH in valid_moves and FixedWindowParserHybrid.zero_cost_sh(
        config, gold_heads
    ):
        moves.append(ArcHybridParser.SH)
    if ArcHybridParser.LA in valid_moves and FixedWindowParserHybrid.zero_cost_la(
        config, gold_heads
    ):
        moves.append(ArcHybridParser.LA)
    if ArcHybridParser.RA in valid_moves and FixedWindowParserHybrid.zero_cost_ra(
        config, gold_heads
    ):
        moves.append(ArcHybridParser.RA)
    return moves


def train_parser_dynamic_oracle(
    train_data: Treebank,
    n_epochs: int = 1,
    batch_size: int = 100,
    lr: float = 1e-2,
) -> FixedWindowParserHybrid:
    # Create the vocabularies
    vocab_words, vocab_tags = make_vocabs(train_data)
    parser = FixedWindowParserHybrid(vocab_words, vocab_tags)
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)

    # Training loop
    for _ in range(n_epochs):
        running_loss = 0
        n_examples = 1
        with tqdm(total=len(train_data)) as pbar:
            batch_iteration = 0
            batch_loss = []
            for sentence in train_data:
                # Separate the words, tags, and heads
                words, tags, heads = zip(*sentence)

                config = parser.initial_config(len(heads))

                # Encode words and tags using the vocabularies
                words = [vocab_words.get(w, UNK_IDX) for w in words]
                tags = [vocab_tags[t] for t in tags]

                while not parser.is_final_config(config):
                    valid_moves = parser.valid_moves(config)
                    features = parser.featurize(words, tags, config)
                    scores = parser.model.forward(features)

                    best_score, predicted_move = float("-inf"), -1
                    for move in valid_moves:
                        if scores[move] > best_score:
                            best_score, predicted_move = scores[move], move

                    zero_cost_moves = dynamic_oracle(config, valid_moves, heads)
                    best_zero_cost_move = max(zero_cost_moves, key=lambda x: scores[x])

                    y = torch.tensor([best_zero_cost_move]).long()

                    loss = F.cross_entropy(scores.unsqueeze(0), y)
                    batch_loss.append(loss)
                    running_loss += loss.item()  # tqdm

                    if predicted_move in zero_cost_moves:
                        config = parser.next_config(config, predicted_move)
                    else:
                        config = parser.next_config(config, best_zero_cost_move)

                    pbar.set_postfix(
                        loss=running_loss / n_examples, run=n_examples
                    )  # tqdm
                    n_examples += 1
                    batch_iteration += 1

                    if batch_iteration == batch_size:
                        optimizer.zero_grad()
                        loss = sum(batch_loss)
                        loss.backward()
                        optimizer.step()
                        # re-init
                        batch_loss = []
                        batch_iteration = 0

                pbar.update(1)

    return parser


def get_uas(
    parser: FixedWindowParser | FixedWindowParserHybrid, gold_sentences: Treebank
) -> float:
    correct = 0
    total = 0
    for sentence in gold_sentences:
        sentence = cast(list[tuple[str, str, int]], sentence)
        zipped_object: tuple[list[str], list[str, list[int]]] = zip(*sentence)
        words, tags, gold_heads = zipped_object
        pred_heads = parser.predict(words, tags)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):  # ignore the pseudo-root
            correct += int(gold == pred)
            total += 1
    return correct / total
