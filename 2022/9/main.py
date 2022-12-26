"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/9

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def are_neigh(a, b):
    return tf.math.less_equal(tf.norm(a - b, ord=tf.experimental.numpy.inf), 1)


# integers to naturals
def to_natural(z):
    if tf.greater_equal(z, 0):
        return tf.cast(2 * z, tf.int64)
    return tf.cast(-2 * z - 1, tf.int64)


def pairing_fn(i, j):
    # https://en.wikipedia.org/wiki/Pairing_function#Hopcroft_and_Ullman_pairing_function

    i, j = to_natural(i), to_natural(j)
    return (i + j) * (i + j + 1) // 2 + j


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = (
        tf.data.TextLineDataset(input_path.as_posix())
        .map(lambda line: tf.strings.split(line, " "))
        .map(lambda pair: (pair[0], tf.strings.to_number(pair[1], tf.int64)))
    )
    head = tf.Variable(tf.zeros((2,), dtype=tf.int64))
    tail = tf.Variable(tf.zeros_like(head))
    pos = tf.lookup.experimental.MutableHashTable(tf.int64, tf.int64, (-1, 0, 0))
    pos.insert([pairing_fn(0, 0)], [(1, 0, 0)])

    def play(direction, amount):

        sign = tf.constant(-1, tf.int64)
        if tf.logical_or(tf.equal(direction, "U"), tf.equal(direction, "R")):
            sign = tf.constant(1, tf.int64)

        axis = tf.constant((0, 1), tf.int64)
        if tf.logical_or(tf.equal(direction, "R"), tf.equal(direction, "L")):
            axis = tf.constant((1, 0), tf.int64)

        for _ in tf.range(amount):
            head.assign_add(sign * axis)
            if tf.logical_not(are_neigh(head, tail)):
                distance = head - tail
                x = tf.math.sign(distance[0])
                y = tf.math.sign(distance[1])

                tail.assign_add(tf.stack([x, y]))

                mapped = pairing_fn(tail[0], tail[1])
                info = pos.lookup([mapped])[0]

                visited, first_coord, second_coord = info[0], info[1], info[2]
                if tf.equal(visited, -1):
                    # first time visited
                    pos.insert(
                        [mapped],
                        [tf.stack([tf.constant(1, tf.int64), tail[0], tail[1]])],
                    )

            tf.print("H: ", head)
            tf.print("T: ", tail)

        return head

    # less than 5888

    dataset = dataset.map(play)

    list(dataset)

    # (#visisted, x, y), where # visisted = 1 visited x,y # visisted = 2 visisted x,y and y,x
    tail_positions = pos.export()[1]
    # tf.print(tf.shape(tail_positions))
    visited_count = tf.reduce_sum(tail_positions[:, 0])

    # tf.print(pos.export()[1], summarize=-1)
    tf.print("part 1: ", visited_count)

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
