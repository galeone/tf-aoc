#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/5
"""

import sys
from typing import Tuple

import tensorflow as tf


class Grid(tf.Module):
    """Grid module. Draws over the grid the various lines and when called
    returns the number of intersection, depending on the puzzle part required.
    """

    def __init__(self, dataset):
        super().__init__()
        bbox_w = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[0], p2[0])))) + 1
        bbox_h = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[1], p2[1])))) + 1
        self._grid = tf.Variable(
            tf.zeros((bbox_w, bbox_h), dtype=tf.int64), trainable=False
        )
        self._dataset = dataset

    @staticmethod
    @tf.function
    def interpolate(p1: tf.Tensor, p2: tf.Tensor):
        """Linear interpolation from p1 to p2 in the discrete 2D grid.
        Args:
            p1: Tensor with values (x, y)
            p2: Tensor with values (x, y)
        Returns:
            The linear interpolation in the discrete 2D grid.
        """
        # +1 handles the case of p1 - p2 == 1
        norm = tf.norm(tf.cast(p1 - p2, tf.float32), ord=tf.experimental.numpy.inf) + 1
        return tf.cast(
            tf.math.ceil(tf.linspace(p1, p2, tf.cast(norm, tf.int64))), tf.int64
        )

    @tf.function
    def __call__(self, part_one: tf.Tensor) -> tf.Tensor:
        """Given the required puzzle part, changes the line drawing on the grid
        and the intersection couunt.
        Args:
            part_one: boolean tensor. When true, only consider straight lines and
                      a threshold of 1. When false, consider straight lines and diagonal
                      lines.
        Returns
            the number of intersections
        """
        self._grid.assign(tf.zeros_like(self._grid))

        for start, end in self._dataset:
            # Discrete interpolation between start and end
            # part 1 requires to consider only straight lines
            # (x1 = x2 or y1 = y2)
            # but I guess (hope) doing the generic discrete interpolation
            # will simplify part 2 (no idea, just a guess)
            float_start = tf.cast(start, tf.float32)
            float_end = tf.cast(end, tf.float32)
            direction = float_start - float_end
            angle = (
                tf.math.atan2(direction[1], direction[0])
                * 180
                / tf.experimental.numpy.pi
            )
            if tf.less(angle, 0):
                angle = 360 + angle
            if tf.logical_or(
                tf.logical_and(
                    tf.logical_not(part_one),
                    tf.logical_and(
                        tf.logical_not(tf.equal(tf.math.mod(angle, 90), 0)),
                        tf.equal(tf.math.mod(angle, 45), 0),
                    ),
                ),
                tf.logical_or(
                    tf.equal(start[0], end[0]),
                    tf.equal(start[1], end[1]),
                ),
            ):
                pixels = self.interpolate(start, end)
                self._grid.assign(
                    tf.tensor_scatter_nd_add(
                        self._grid, pixels, tf.ones(tf.shape(pixels)[0], dtype=tf.int64)
                    )
                )

        # tf.print(tf.transpose(grid, perm=(1, 0)), summarize=-1)
        threshold = tf.constant(1, tf.int64)
        mask = tf.greater(self._grid, threshold)
        return tf.reduce_sum(tf.cast(mask, tf.int64))


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    def _get_segment(
        line: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        points = tf.strings.split(line, " -> ")
        p1 = tf.strings.split(points[0], ",")
        p2 = tf.strings.split(points[1], ",")

        x1 = tf.strings.to_number(p1[0], tf.int64)
        y1 = tf.strings.to_number(p1[1], tf.int64)
        x2 = tf.strings.to_number(p2[0], tf.int64)
        y2 = tf.strings.to_number(p2[1], tf.int64)
        return tf.convert_to_tensor((x1, y1)), tf.convert_to_tensor((x2, y2))

    dataset = tf.data.TextLineDataset("input").map(_get_segment)
    grid = Grid(dataset)

    tf.print("# overlaps (part one): ", grid(tf.constant(True)))
    tf.print("# overlaps (part two): ", grid(tf.constant(False)))


if __name__ == "__main__":
    sys.exit(main())
