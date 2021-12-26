#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/11
"""

import sys
from typing import Tuple

import tensorflow as tf


class FlashCounter(tf.Module):
    def __init__(self, population, steps):
        super().__init__()

        self._steps = steps
        self._population = tf.Variable(population, dtype=tf.int64)
        self._counter = tf.Variable(0, dtype=tf.int64)

        self._zero = tf.constant(0, dtype=tf.int64)
        self._one = tf.constant(1, dtype=tf.int64)
        self._nine = tf.constant(9, tf.int64)
        self._ten = tf.constant(10, dtype=tf.int64)

        self._queue = tf.queue.FIFOQueue(-1, [tf.int64])

        self._flashmap = tf.Variable(tf.zeros_like(self._population))

    @tf.function
    def _neighs(
        self, grid: tf.Tensor, center: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        y, x = center[0], center[1]

        shape = tf.shape(grid, tf.int64) - 1

        if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):  # 0,0
            mask = tf.constant([(1, 0), (0, 1), (1, 1)])
        elif tf.logical_and(tf.equal(y, shape[0]), tf.equal(x, shape[1])):  # h,w
            mask = tf.constant([(-1, 0), (0, -1), (-1, -1)])
        elif tf.logical_and(tf.less(y, 1), tf.equal(x, shape[1])):  # top right
            mask = tf.constant([(0, -1), (1, 0), (1, -1)])
        elif tf.logical_and(tf.less(x, 1), tf.equal(y, shape[0])):  # bottom left
            mask = tf.constant([(-1, 0), (-1, 1), (0, 1)])
        elif tf.less(x, 1):  # left
            mask = tf.constant([(1, 0), (-1, 0), (-1, 1), (0, 1), (1, 1)])
        elif tf.equal(x, shape[0]):  # right
            mask = tf.constant([(-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)])
        elif tf.less(y, 1):  # top
            mask = tf.constant([(0, -1), (0, 1), (1, 0), (1, -1), (1, 1)])
        elif tf.equal(y, shape[1]):  # bottom
            mask = tf.constant([(0, -1), (0, 1), (-1, 0), (-1, -1), (-1, 1)])
        else:  # generic
            mask = tf.constant(
                [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
            )

        # tf.print(center, mask, summarize=-1)

        coords = center + tf.cast(mask, tf.int64)
        neighborhood = tf.gather_nd(grid, coords)
        return neighborhood, coords

    @tf.function
    def __call__(self):
        for step in tf.range(self._steps):
            # First, the energy level of each octopus increases by 1.
            self._population.assign_add(tf.ones_like(self._population))

            # Then, any octopus with an energy level greater than 9 flashes.
            flashing_coords = tf.where(tf.greater(self._population, self._nine))
            self._queue.enqueue_many(flashing_coords)

            # This increases the energy level of all adjacent octopuses by 1, including octopuses that are diagonally adjacent.
            # If this causes an octopus to have an energy level greater than 9, it also flashes.
            # This process continues as long as new octopuses keep having their energy level increased beyond 9.
            # (An octopus can only flash at most once per step.)
            while tf.greater(self._queue.size(), 0):
                p = self._queue.dequeue()
                if tf.greater(self._flashmap[p[0], p[1]], 0):
                    continue
                self._flashmap.scatter_nd_update([p], [1])

                _, neighs_coords = self._neighs(self._population, p)
                updates = tf.repeat(
                    self._one,
                    tf.shape(neighs_coords, tf.int64)[0],
                )
                self._population.scatter_nd_add(neighs_coords, updates)
                flashing_coords = tf.where(tf.greater(self._population, self._nine))
                self._queue.enqueue_many(flashing_coords)

            # Finally, any octopus that flashed during this step has its energy level set to 0, as it used all of its energy to flash.
            indices = tf.where(tf.equal(self._flashmap, self._one))
            if tf.greater(tf.size(indices), 0):
                shape = tf.shape(indices, tf.int64)
                updates = tf.repeat(self._zero, shape[0])
                self._counter.assign_add(shape[0])
                self._population.scatter_nd_update(indices, updates)

            self._flashmap.assign(tf.zeros_like(self._flashmap))

            # tf.print(step, self._population, summarize=-1)
        return self._counter


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    population = tf.convert_to_tensor(
        list(
            tf.data.TextLineDataset("input")
            .map(tf.strings.bytes_split)
            .map(lambda string: tf.strings.to_number(string, out_type=tf.int64))
        )
    )

    steps = tf.constant(100, tf.int64)
    flash_counter = FlashCounter(population, steps)
    tf.print(flash_counter())


if __name__ == "__main__":
    sys.exit(main())
