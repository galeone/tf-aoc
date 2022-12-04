"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/4

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    pairs = dataset.map(lambda line: tf.strings.split(line, ","))
    ranges = pairs.map(
        lambda pair: tf.strings.to_number(tf.strings.split(pair, "-"), tf.int64)
    )

    contained = ranges.filter(
        lambda pair: tf.logical_or(
            tf.logical_and(
                tf.math.less_equal(pair[0][0], pair[1][0]),
                tf.math.greater_equal(pair[0][1], pair[1][1]),
            ),
            tf.logical_and(
                tf.math.less_equal(pair[1][0], pair[0][0]),
                tf.math.greater_equal(pair[1][1], pair[0][1]),
            ),
        )
    )
    contained_tensor = tf.convert_to_tensor(
        list(iter(contained.map(lambda ragged: tf.sparse.to_dense(ragged.to_sparse()))))
    )
    tf.print("Fully contained ranges: ", tf.shape(contained_tensor)[0])

    overlapping = ranges.filter(
        lambda pair: tf.logical_not(
            tf.logical_or(
                tf.math.less(pair[0][1], pair[1][0]),
                tf.math.less(pair[1][1], pair[0][0]),
            )
        )
    )

    overlapping_tensor = tf.convert_to_tensor(
        list(
            iter(overlapping.map(lambda ragged: tf.sparse.to_dense(ragged.to_sparse())))
        )
    )

    tf.print("Overlapping ranges: ", tf.shape(overlapping_tensor)[0])
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
