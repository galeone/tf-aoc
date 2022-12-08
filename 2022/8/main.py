"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/1

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    dataset = dataset.map(lambda line: tf.strings.bytes_split(line))
    dataset = dataset.map(lambda x: tf.strings.to_number(x, tf.int64))

    grid = tf.Variable(list(dataset.as_numpy_iterator()))

    visibles = tf.Variable(0, dtype=tf.int64)
    # edges
    grid_shape = tf.shape(grid, tf.int64)
    visibles.assign_add(tf.reduce_sum(grid_shape * 2) - 4)

    # inner
    for col in tf.range(1, grid_shape[0] - 1):
        for row in tf.range(1, grid_shape[1] - 1):
            x = grid[col, row]

            visible_right = tf.reduce_all(x > grid[col, row + 1 :])
            if visible_right:
                visibles.assign_add(1)
                continue
            visible_left = tf.reduce_all(x > grid[col, :row])
            if visible_left:
                visibles.assign_add(1)
                continue

            visible_bottom = tf.reduce_all(x > grid[col + 1 :, row])
            if visible_bottom:
                visibles.assign_add(1)
                continue
            visible_top = tf.reduce_all(x > grid[:col, row])
            if visible_top:
                visibles.assign_add(1)
                continue

    tf.print("part 1: ", visibles)

    scenic_score = tf.Variable(0, tf.int64)  #  t * l * b * r
    t = tf.Variable(0, tf.int64)
    l = tf.Variable(0, tf.int64)
    b = tf.Variable(0, tf.int64)
    r = tf.Variable(0, tf.int64)
    for col in tf.range(1, grid_shape[0] - 1):
        for row in tf.range(1, grid_shape[1] - 1):
            x = grid[col, row]
            views = grid - x

            right = views[col, row + 1 :]
            # the loop is left to right
            left = tf.reverse(views[col, :row], axis=[0])
            # the loop is bottom to top
            top = tf.reverse(views[:col, row], axis=[0])
            bottom = views[col + 1 :, row]

            for tree in right:
                r.assign_add(1)
                if tf.greater_equal(tree, 0):
                    break
            for tree in left:
                l.assign_add(1)
                if tf.greater_equal(tree, 0):
                    break
            for tree in bottom:
                b.assign_add(1)
                if tf.greater_equal(tree, 0):
                    break
            for tree in top:
                t.assign_add(1)
                if tf.greater_equal(tree, 0):
                    break
            scenic_node = t * l * b * r
            if tf.greater(scenic_node, scenic_score):
                scenic_score.assign(scenic_node)
            r.assign(0)
            l.assign(0)
            t.assign(0)
            b.assign(0)

    tf.print("part 2: ", scenic_score)
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
