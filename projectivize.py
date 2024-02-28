# Projectivize trees in the CoNLL-X format using lifting
# Usage: python projectivize.py < CONLL > CONLL

# Bonus: Count the number of projective trees
from collections.abc import Iterable
from typing import TextIO


def trees(fp: TextIO) -> Iterable[list[str]]:
    buffer: list[str] = []
    for line in fp:
        stripped_line = line.rstrip()  # strip off the trailing newline
        if not stripped_line.startswith("#"):
            if len(stripped_line) == 0:
                yield buffer
                buffer = []
            else:
                columns = stripped_line.split("\t")
                if columns[0].isdigit():  # skip range tokens
                    buffer.append(columns)


def heads(rows: list[str]) -> list[int]:
    return [0] + [int(row[6]) for row in rows]


DN = +1
SH = 0
UP = -1


def traverse(heads: list[int]) -> Iterable[tuple[int, int]]:
    marked: list[bool] = [False] * len(heads)
    cursor: int = 0
    marked[cursor] = True
    for i in range(len(heads)):
        bend = i
        path = []
        while not marked[bend]:
            path.append(bend)
            bend = heads[bend]
        while cursor != bend:
            yield cursor, UP
            marked[cursor] = False
            cursor = heads[cursor]
        while len(path) > 0:
            cursor = path.pop()
            marked[cursor] = True
            yield cursor, DN
        yield cursor, SH
        assert cursor == i
    while cursor != 0:
        yield cursor, UP
        marked[cursor] = False
        cursor = heads[cursor]


def is_projective(heads: list[int]) -> bool:
    seen = [False] * len(heads)
    for cursor, d in traverse(heads):
        if d == DN:
            if seen[cursor]:
                return False
            else:
                seen[cursor] = True
    return True


def projectivize(heads: list[int]) -> list[int]:
    pheads = [0] * len(heads)
    dangling = [[] for _ in heads]
    head_blk = [False] * len(heads)
    for cursor, d in traverse(heads):
        if d == UP:
            if head_blk[heads[cursor]]:
                for node in dangling[cursor]:
                    pheads[node] = heads[cursor]
            else:
                dangling[heads[cursor]] += dangling[cursor]
            dangling[cursor] = []
            head_blk[cursor] = False
        if d == SH:
            head_blk[cursor] = True
            for node in dangling[cursor]:
                pheads[node] = cursor
            dangling[cursor] = [cursor]
    return pheads


def projectivized_trees(fp: TextIO) -> Iterable[list[str]]:
    for tree in trees(fp):
        pheads = projectivize(heads(tree))
        for i, row in enumerate(tree):
            row[6] = "%d" % pheads[i + 1]
        yield tree


def emit(tree: list[str]) -> None:
    for row in tree:
        print("\t".join(row))
    print()


def cmd_count_projective() -> None:
    import sys

    k = 0
    n = 0
    for tree in trees(sys.stdin):
        k += is_projective(heads(tree))
        n += 1
    print(f"{k / n:.2%}")


def cmd_projectivize() -> None:
    import sys

    for ptree in projectivized_trees(sys.stdin):
        emit(ptree)


def filename_projectivize(filename: str, target_filename: str) -> None:
    print(f"Projectivizing {filename} to {target_filename}")
    with open(target_filename, "w") as file_out:
        with open(filename) as file_in:
            for ptree in projectivized_trees(file_in):
                for row in ptree:
                    file_out.write("\t".join(row) + "\n")
                file_out.write("\n")


if __name__ == "__main__":
    # cmd_count_projective()
    import sys

    if len(sys.argv) > 2:
        filename_projectivize(sys.argv[1], sys.argv[2])
