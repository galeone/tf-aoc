"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/10

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    cycle = tf.Variable(0, dtype=tf.int32)
    X = tf.Variable(1, dtype=tf.int32)

    lut = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(["noop", "addx"]), tf.constant([0, 1])
        ),
        default_value=-1,
    )

    dataset = tf.data.TextLineDataset(input_path.as_posix())

    dataset = dataset.map(lambda line: tf.strings.split(line, " "))

    @tf.function
    def opval(pair):
        if tf.equal(tf.shape(pair)[0], 1):
            return pair[0], tf.constant(0, tf.int32)

        return pair[0], tf.strings.to_number(pair[1], tf.int32)

    dataset = dataset.map(opval)

    noop_id = lut.lookup(tf.constant(["noop"]))[0]
    noop = tf.stack((noop_id, 0), axis=0)
    invalid = tf.constant((-1, -1))

    @tf.function
    def prepend_noop(op, val):
        if tf.equal(op, "noop"):
            return tf.stack([noop, invalid], axis=0)

        return tf.stack(
            [
                noop,
                tf.stack((lut.lookup(tf.expand_dims(op, axis=0))[0], val), axis=0),
            ],
            axis=0,
        )

    dataset = (
        dataset.map(prepend_noop)
        .unbatch()
        .filter(lambda op_val: tf.not_equal(op_val[0], -1))  # remove invalid
        .map(lambda op_val: (op_val[0], op_val[1]))
    )
    # now every element in the dataset is a clock cycle

    prevX = tf.Variable(X)

    def clock(op, val):
        prevX.assign(X)
        if tf.equal(op, noop_id):
            pass
        else:  # addx
            X.assign_add(val)

        cycle.assign_add(1)

        if tf.reduce_any([tf.equal(cycle, value) for value in range(20, 221, 40)]):
            return [cycle, prevX, prevX * cycle]
        return [cycle, prevX, -1]

    dataset = dataset.map(clock).filter(
        lambda c, x, strenght: tf.not_equal(strenght, -1)
    )

    strenghts = tf.convert_to_tensor((list(dataset.as_numpy_iterator())))

    sumsix = tf.reduce_sum(strenghts[:, -1])
    tf.print("Sum of six signal strenght: ", sumsix)

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
