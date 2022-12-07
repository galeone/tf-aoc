"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/6

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())

    chars = tf.convert_to_tensor(
        next(
            dataset.map(lambda line: tf.strings.bytes_split(line))
            .take(1)
            .as_numpy_iterator()
        )
    )

    tf.print(chars)

    dataset = tf.data.Dataset.from_tensors(tf.reshape(chars, [-1, 1])).unbatch()

    interleaved = tf.data.Dataset.range(4).interleave(
        lambda offset: dataset.skip(offset).batch(4),
        cycle_length=4,
        block_length=1,
        num_parallel_calls=4,
        deterministic=True,
    )

    for count, b in enumerate(interleaved):
        y, _ = tf.unique(tf.reshape(b, -1))
        if tf.equal(tf.shape(y)[0], 4):
            tf.print(y)
            # 1: starts from 0
            # 3: the remaining chars in the sequence
            tf.print("unique found at char: ", count + 4)
            break

    # identical, just range over 14 interleaved datasets with
    # a batch of 14
    interleaved = tf.data.Dataset.range(14).interleave(
        lambda offset: dataset.skip(offset).batch(14),
        cycle_length=14,
        block_length=1,
        num_parallel_calls=14,
        deterministic=True,
    )

    for count, b in enumerate(interleaved):
        y, _ = tf.unique(tf.reshape(b, -1))
        if tf.equal(tf.shape(y)[0], 14):
            tf.print(y)
            # 1: starts from 0
            # 13: the remaining chars in the sequence
            tf.print("unique 14 chars found after reading : ", count + 14, " chars")
            break

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
