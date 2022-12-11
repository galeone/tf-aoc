"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/1

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def are_neigh(a, b):
    return tf.math.less_equal(tf.norm((a - b), ord=tf.experimental.numpy.inf), 1)


# integers to naturals
def to_natural(z):
    if tf.greater_equal(z, 0):
        return tf.cast(2 * z, tf.int32)
    return tf.cast(-2 * z - 1, tf.int32)


def pairing_fn(i, j):
    # https://en.wikipedia.org/wiki/Pairing_function#Hopcroft_and_Ullman_pairing_function

    i, j = to_natural(i), to_natural(j)
    return (i + j) * (i + j + 1) // 2 + j


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = (
        tf.data.TextLineDataset(input_path.as_posix())
        .map(lambda line: tf.strings.split(line, " "))
        .map(lambda pair: (pair[0], tf.strings.to_number(pair[1], tf.int32)))
    )
    head = tf.Variable(tf.zeros((2,), dtype=tf.int32))
    tail = tf.Variable(tf.zeros_like(head))
    pos = tf.lookup.experimental.MutableHashTable(tf.int32, tf.int32, (0, 0, 0))
    pos.insert([pairing_fn(0, 0)], [(1, 0, 0)])

    def play(direction, amount):
        tf.print(direction, amount)

        sign = -1
        if tf.logical_or(tf.equal(direction, "U"), tf.equal(direction, "R")):
            sign = 1

        axis = (0, 1)
        if tf.logical_or(tf.equal(direction, "R"), tf.equal(direction, "L")):
            axis = (1, 0)

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
                if tf.equal(visited, 0):
                    # first time visited
                    pos.insert([mapped], [(1, tail[0], tail[1])])
                elif tf.reduce_all(
                    [
                        tf.equal(visited, 1),
                        tf.equal(first_coord, tail[1]),
                        tf.equal(second_coord, tail[0]),
                    ]
                ):
                    # mapped to the same number, but different cords (e.g [3, 2] and [2, 3] both mapped to 6)
                    pos.insert([mapped], [(2, tail[1], tail[0])])

            tf.print("H: ", head)
            tf.print("T: ", tail)

        return head

    # less than 5888

    dataset = dataset.map(play)

    for _ in dataset:
        pass

    # (#visisted, x, y), where # visisted = 1 visited x,y # visisted = 2 visisted x,y and y,x
    tail_positions = pos.export()[1]
    # tf.print(tf.shape(tail_positions))
    visited_count = tf.reduce_sum(tail_positions[:, 0])

    # tf.print(pos.export()[1], summarize=-1)
    tf.print("part 1: ", visited_count)

    """
    draw = tf.Variable(
        tf.zeros(
            (
                tf.reduce_max(tail_positions[:, 1]) + 1,
                tf.reduce_max(tail_positions[:, 2]) + 1,
            ),
            dtype=tf.int32,
        )
    )
    indices = tail_positions[:, 1:]
    flipped_indices = tf.reverse_sequence(
        indices, tf.repeat(2, tf.shape(indices)[0]), seq_axis=1, batch_axis=0
    )

    # tf.print(indices, summarize=-1)
    # tf.print(flipped_indices, summarize=-1)
    update = tf.tensor_scatter_nd_add(
        draw,
        flipped_indices,
        tf.repeat(1, tf.shape(indices)[0]),
    )
    tf.print(update, summarize=-1)
    """

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
