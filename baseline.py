import os

import torch

from parsing import (
    FixedWindowParser,
    FixedWindowParserHybrid,
    get_uas,
    train_parser,
    train_parser_dynamic_oracle,
)
from tagging import FixedWindowTagger, accuracy, train_tagger
from treebank import Treebank

torch.manual_seed(12345)


def evaluate(
    tagger: FixedWindowTagger,
    parser: FixedWindowParser | FixedWindowParserHybrid,
    gold_sentences: Treebank,
) -> tuple[float, float]:
    correct_tagger = 0
    total_tagger = 0
    correct_parser = 0
    total_parser = 0
    for sentence in gold_sentences:
        words, gold_tags, gold_heads = zip(*sentence)
        pred_tags = tagger.predict(words)
        for gold, pred in zip(gold_tags[1:], pred_tags[1:]):
            correct_tagger += int(gold == pred)
            total_tagger += 1
        pred_heads = parser.predict(words, pred_tags)
        for gold, pred in zip(gold_heads[1:], pred_heads[1:]):
            correct_parser += int(gold == pred)
            total_parser += 1
    return correct_tagger / total_tagger, correct_parser / total_parser


if __name__ == "__main__":
    import sys

    AVAILABLE_TREEBANKS = ["en_ewt", "ja_gsd"]
    AVAILABLE_PARSER_TYPES = ["arc-standard", "arc-hybrid"]
    TREEBANK_ROOT = "treebank"

    if len(sys.argv) > 1:
        # If we have been given a filename, we will projectivize it
        arguments = sys.argv[1:]
        for bank_id in arguments:
            if bank_id not in AVAILABLE_TREEBANKS:
                print(
                    f"Unknown treebank: {bank_id}, must be one of {AVAILABLE_TREEBANKS}"
                )
                sys.exit(1)
        treebanks = arguments
    else:
        treebanks = AVAILABLE_TREEBANKS

    print(f"Using treebanks: {treebanks}")

    for treebank in treebanks:
        print(f"Current treebank: {treebank}")
        train_data_filename = f"{TREEBANK_ROOT}/{treebank}/{treebank}-ud-train.conllu"
        dev_data_filename = f"{TREEBANK_ROOT}/{treebank}/{treebank}-ud-dev.conllu"

        train_data = Treebank.from_non_projective(train_data_filename)
        dev_data = Treebank.from_non_projective(dev_data_filename)

        # delete tmp file
        os.remove("tmp")

        print("Training tagger")
        tagger = train_tagger(train_data)
        print(f"Tagging accuracy: {accuracy(tagger, dev_data):.4f}")

        for parser_type in AVAILABLE_PARSER_TYPES:
            for use_dynamic_oracle in [False, True]:
                print(f"Current parser type: {parser_type}")
                print(f"Using dynamic oracle: {use_dynamic_oracle}")
                if use_dynamic_oracle and parser_type == "arc-standard":
                    continue
                elif use_dynamic_oracle and parser_type == "arc-hybrid":
                    parser = train_parser_dynamic_oracle(train_data, n_epochs=1)
                else:
                    parser = train_parser(
                        train_data, parser_type=parser_type, n_epochs=1
                    )
                print(f"Gold uas: {get_uas(parser, dev_data):.4f}")
                acc, uas = evaluate(tagger, parser, dev_data)
                print(f"Tagging accuracy: {acc:.4f}, Retagged uas: {uas:.4f}")
