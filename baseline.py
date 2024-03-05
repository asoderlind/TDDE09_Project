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

# complete .to(device) to all relevant functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

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

        for parser_type in AVAILABLE_PARSER_TYPES:
            for use_dynamic_oracle in [False, True]:
                if use_dynamic_oracle and parser_type == "arc-standard":
                    continue
                elif use_dynamic_oracle and parser_type == "arc-hybrid":
                    parser = train_parser_dynamic_oracle(
                        train_data, n_epochs=1, batch_size=100
                    )
                else:
                    parser = train_parser(
                        train_data, parser_type=parser_type, n_epochs=1
                    )
                golden_uas = get_uas(parser, dev_data)
                tagger_acc, retag_uas = evaluate(tagger, parser, dev_data)
                oracle = "dynamic" if use_dynamic_oracle else "static"
                print("Parsing system\tOracle\tTagging acc.\t UAS")
                print(f"{parser_type}\t {oracle}\t (Golden) \t {golden_uas:.4f}")
                print(
                    f"{parser_type}\t {oracle}\t {tagger_acc:.4f}\t\t {retag_uas:.4f}"
                )
