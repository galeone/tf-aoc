"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/12

of the Advent of Code 2022.
"""

import string
import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    dataset = dataset.map(tf.strings.bytes_split)

    keys_tensor = tf.concat(
        [tf.strings.bytes_split(string.ascii_lowercase), tf.constant(["S", "E"])],
        axis=0,
    )
    values_tensor = tf.concat([tf.range(0, 26), tf.constant([-1, 26])], axis=0)
    lut = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
        default_value=-1,
    )

    dataset = dataset.map(lut.lookup)

    grid = tf.convert_to_tensor(list(dataset))
    visited = tf.Variable(tf.zeros_like(grid))

    @tf.function
    def _neighs(grid: tf.Tensor, center: tf.Tensor):
        y, x = center[0], center[1]

        shape = tf.shape(grid) - 1

        if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):  # 0,0
            mask = tf.constant([(1, 0), (0, 1)])
        elif tf.logical_and(tf.equal(y, shape[0]), tf.equal(x, shape[1])):  # h,w
            mask = tf.constant([(-1, 0), (0, -1)])
        elif tf.logical_and(tf.less(y, 1), tf.equal(x, shape[1])):  # top right
            mask = tf.constant([(0, -1), (1, 0)])
        elif tf.logical_and(tf.less(x, 1), tf.equal(y, shape[0])):  # bottom left
            mask = tf.constant([(-1, 0), (0, 1)])
        elif tf.less(x, 1):  # left
            mask = tf.constant([(1, 0), (-1, 0), (0, 1)])
        elif tf.equal(x, shape[1]):  # right
            mask = tf.constant([(-1, 0), (1, 0), (0, -1)])
        elif tf.less(y, 1):  # top
            mask = tf.constant([(0, -1), (0, 1), (1, 0)])
        elif tf.equal(y, shape[0]):  # bottom
            mask = tf.constant([(0, -1), (0, 1), (-1, 0)])
        else:  # generic
            mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1)])

        coords = center + mask
        neighborhood = tf.gather_nd(grid, coords)
        return neighborhood, coords

    queue = tf.queue.FIFOQueue(
        tf.cast(tf.reduce_prod(tf.shape(grid)), tf.int32),
        tf.int32,
        (3,),  # x,y,distance
    )

    ends = tf.cast(tf.where(tf.greater_equal(grid, 25)), tf.int32)
    start = tf.cast(tf.where(tf.equal(grid, -1))[0], tf.int32)

    def bfs(part2=tf.constant(False)):
        if tf.logical_not(part2):
            queue.enqueue(tf.concat([start, tf.constant([0])], axis=0))
            dest_val = 25

            def condition(n_vals, me_val):
                return tf.where(tf.less_equal(n_vals, me_val + 1))

        else:
            end = tf.cast(tf.where(tf.equal(grid, 26)), tf.int32)[0]
            queue.enqueue(tf.concat([end, tf.constant([0])], axis=0))
            dest_val = 1

            def condition(n_vals, me_val):
                return tf.where(tf.greater_equal(n_vals, me_val - 1))

        while tf.greater(queue.size(), 0):
            v = queue.dequeue()
            me, distance = v[:2], v[2]
            me_val = tf.gather_nd(grid, [me])
            already_visited = tf.squeeze(tf.cast(tf.gather_nd(visited, [me]), tf.bool))
            if tf.logical_not(already_visited):
                if tf.reduce_all(tf.equal(me_val, dest_val)):
                    return distance - 1
                visited.assign(tf.tensor_scatter_nd_add(visited, [me], [1]))

                n_vals, n_coords = _neighs(grid, me)
                potential_dests = tf.gather_nd(
                    n_coords,
                    condition(n_vals, me_val),
                )

                not_visited = tf.equal(tf.gather_nd(visited, potential_dests), 0)
                neigh_not_visited = tf.gather_nd(potential_dests, tf.where(not_visited))

                to_visit = tf.concat(
                    [
                        neigh_not_visited,
                        tf.reshape(
                            tf.repeat(distance + 1, tf.shape(neigh_not_visited)[0]),
                            (-1, 1),
                        ),
                    ],
                    axis=1,
                )
                queue.enqueue_many(to_visit)

        return -1

    tf.print("Steps: ", bfs())
    queue.dequeue_many(queue.size())
    visited.assign(tf.zeros_like(visited))

    tf.print("Part 2: ", bfs(True))
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
