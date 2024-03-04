from treebank import Treebank

PAD = "[PAD]"
UNK = "[UNK]"

SH, LA, RA = 0, 1, 2

PAD_IDX = 0
UNK_IDX = 1


def make_vocabs(gold_data: Treebank) -> tuple[dict[str, int], dict[str, int]]:
    vocab_words = {PAD: PAD_IDX, UNK: UNK_IDX}
    vocab_tags = {PAD: PAD_IDX}
    for sentence in gold_data:
        for word, tag, _ in sentence:
            if word not in vocab_words:
                vocab_words[word] = len(vocab_words)
            if tag not in vocab_tags:
                vocab_tags[tag] = len(vocab_tags)
    return vocab_words, vocab_tags
